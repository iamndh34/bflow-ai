"""
COA Agent - Corrective RAG Architecture với LangGraph

Theo structure của idea.md:
1. STATE DEFINITION: CorrectiveRAGState
2. 4 NODES: retrieve, generate_draft, grade_answer, rewrite
3. StateGraph with conditional routing
4. Loop max 2 lần rewrite
"""
import re
import logging
from typing import Literal, Dict, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..services.coa_index import get_coa_index
from ..core.ollama_client import get_ollama_client
from ..core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# BƯỚC 1: STATE DEFINITION (CorrectiveRAGState)
# =============================================================================

class CorrectiveRAGState(TypedDict):
    """
    State cho Corrective RAG workflow

    Theo idea.md structure:
    - messages: List[BaseMessage] - Lịch sử chat
    - query: str - Query gốc
    - rewritten_query: str - Query đã được rewrite
    - documents: List[str] - Documents retrieved từ COA
    - answer: str - Câu trả lời generated
    - confidence: float - Confidence score (0-1)
    - retry_count: int - Số lần đã retry/rewrite
    - needs_rewrite: bool - Có cần rewrite query không
    """
    messages: List[dict]
    query: str
    rewritten_query: str
    documents: List[str]
    answer: str
    confidence: float
    retry_count: int
    needs_rewrite: bool


# =============================================================================
# BƯỚC 2: 4 NODES (retrieve, generate_draft, grade_answer, rewrite)
# =============================================================================

def node_retrieve(state: CorrectiveRAGState) -> CorrectiveRAGState:
    """
    NODE RETRIEVE: Lấy tài khoản từ COA index

    Chức năng:
    1. Kiểm tra relevance - câu hỏi có liên quan đến KO không?
    2. Lấy tài khoản theo code (ưu tiên theo question)
    3. Fallback: Tìm theo keyword nếu không có mã

    Args:
        state: State hiện tại

    Returns:
        State cập nhật với documents (hoặc answer nếu không liên quan)
    """
    logger.info("\n" + "="*60)
    logger.info("🔍 NODE RETRIEVE: Lấy tài khoản")
    logger.info("="*60)
    logger.info(f"📥 Query: {state['query']}")

    # Relevance check - phát hiện câu hỏi không liên quan
    query_lower = state['query'].lower()

    # Từ khóa liên quan đến kế toán/tài khoản
    accounting_keywords = [
        'tài khoản', 'tk', 'thông tư', 'tt', 'hạch toán', 'kế toán',
        'có', 'nợ', 'số dư', 'đối tượng', 'phân loại', 'chart', 'account',
        'so sánh', 'khác', 'giữa'
    ]

    # Kiểm tra có từ khóa kế toán không HOẶC có số tài khoản (3-5 chữ số)
    has_accounting_keyword = any(kw in query_lower for kw in accounting_keywords)
    has_account_code = bool(re.search(r'\b\d{3,5}\b', query_lower))

    if not has_accounting_keyword and not has_account_code:
        logger.info(f"⚠️  Query không liên quan đến kế toán/tài khoản")
        # Trả về trực tiếp để kết thúc workflow
        return {
            **state,
            "answer": "Xin lỗi, câu hỏi này không liên quan đến lĩnh vực kế toán/tài khoản. Tôi có thể giúp bạn tìm thông tin về:\n- Tài khoản kế toán (VD: TK 111, TK 156)\n- So sánh tài khoản giữa TT99 và TT200\n- Chức năng và cách hạch toán các tài khoản",
            "confidence": 1.0,
            "needs_rewrite": False,
        }

    # Query hiện tại = rewritten_query nếu có, ngược lại query gốc
    query = state.get("rewritten_query") or state["query"]
    logger.info(f"🔍 Query đang sử dụng: {query}")

    # Bước 1: Extract mã tài khoản từ query
    code_match = re.search(r'\b(\d{3,5})\b', query)
    code = code_match.group(1) if code_match else ""
    logger.info(f"🔍 Mã TK extract được: {code}")

    # Bước 2: Tìm trong COA index
    idx = get_coa_index()

    # Ưu tiên theo code
    use_tt200 = "tt200" in query.lower()
    acc = idx.get_by_code(code, use_tt200=use_tt200) if code else None

    documents = []

    if acc:
        logger.info(f"✅ Tìm thấy: TK {acc['code']} - {acc['name']}")
        documents.append(f"TK {acc['code']}: {acc['name']}")
        documents.append(f"Loại: {acc.get('type_name', 'N/A')}")
        documents.append(f"Chuẩn mực: {'TT200' if use_tt200 else 'TT99'}")
    else:
        # Fallback: Tìm theo keyword
        logger.warning(f"⚠️  Không tìm thấy theo mã, tìm theo keyword...")
        results = idx.search_by_keyword(query, limit=5)
        if results:
            logger.info(f"✅ Tìm thấy {len(results)} TK theo keyword")
            for acc in results[:3]:
                documents.append(f"TK {acc['code']}: {acc['name']}")
        else:
            logger.error(f"❌ Không tìm thấy tài khoản nào")
            documents.append("Không tìm thấy tài khoản phù hợp.")

    return {
        **state,
        "documents": documents,
    }


def node_generate_draft(state: CorrectiveRAGState) -> CorrectiveRAGState:
    """
    NODE GENERATE DRAFT: Sinh câu trả lời từ documents

    Chức năng:
    1. Build prompt với context từ documents
    2. Gọi LLM generate answer
    3. Trả về answer draft

    Args:
        state: State với documents đã retrieve

    Returns:
        State cập nhật với answer
    """
    logger.info("\n" + "="*60)
    logger.info("🤖 NODE GENERATE_DRAFT: Sinh câu trả lời")
    logger.info("="*60)

    query = state.get("rewritten_query") or state["query"]
    documents = state.get("documents", [])

    context = "\n".join(documents) if documents else "Không có thông tin."
    logger.info(f"📝 Context:\n{context}")

    # Tạo prompt
    prompt = f"""Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT.

CÂU HỎI: {query}

THÔNG TIN TÀI KHOẢN:
{context}

Hãy trả lời theo format:
1. THÔNG TIN CƠ BẢN
- Số hiệu
- Tên tài khoản
- Phân loại

2. NỘI DUNG PHẢN ÁNH
[Mô tả chức năng của tài khoản]

3. KẾT CẤU
- Bên Nợ: Ghi nhận gì
- Bên Có: Ghi nhận gì
- Số dư: Thường nằm bên nào

4. LƯU Ý
[Các lưu ý khi hạch toán]

Kết thúc: (Căn cứ: Phụ lục II - Thông tư 99/2025/TT-BTC)
"""

    # Gọi LLM
    logger.info(f"🔄 Đang gọi Ollama...")
    try:
        llm = get_ollama_client()
        response = llm.chat(
            model=settings.GENERATION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            options=settings.OLLAMA_OPTIONS,
            stream=False
        )
        answer = response.get("message", {}).get("content", "")
        logger.info(f"✅ Nhận phản hồi LLM: {len(answer)} ký tự")

    except Exception as e:
        logger.error(f"❌ Lỗi LLM: {e}")
        answer = f"Đã xảy ra lỗi: {str(e)}"

    return {
        **state,
        "answer": answer,
    }


def node_grade_answer(state: CorrectiveRAGState) -> CorrectiveRAGState:
    """
    NODE GRADE ANSWER: Đánh giá chất lượng câu trả lời

    Chức năng:
    1. Kiểm tra answer có hữu ích không
    2. Gán confidence score
    3. Quyết định có cần rewrite query không

    Args:
        state: State với answer đã generate

    Returns:
        State cập nhật với confidence, needs_rewrite
    """
    logger.info("\n" + "="*60)
    logger.info("📊 NODE GRADE_ANSWER: Đánh giá chất lượng")
    logger.info("="*60)

    answer = state.get("answer", "")
    documents = state.get("documents", [])

    # Heuristic grading
    confidence = 0.5  # Default
    needs_rewrite = False

    # Check 1: Answer không empty
    if not answer or len(answer) < 50:
        confidence = 0.2
        needs_rewrite = True
        logger.warning("⚠️  Answer quá ngắn hoặc empty")
    else:
        confidence += 0.3

    # Check 2: Documents có nội dung
    if documents and "Không tìm thấy" not in documents[0]:
        confidence += 0.2
    else:
        confidence -= 0.2
        needs_rewrite = True
        logger.warning("⚠️  Không có documents phù hợp")

    # Check 3: Retry count limit
    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        confidence = max(confidence, 0.6)  # Force accept
        needs_rewrite = False
        logger.info(f"✅ Đạt giới hạn retry ({retry_count}), chấp nhận answer")

    # Cap confidence
    confidence = max(0.0, min(1.0, confidence))

    logger.info(f"📊 Confidence: {confidence:.2f}")
    logger.info(f"📊 Needs rewrite: {needs_rewrite}")

    return {
        **state,
        "confidence": confidence,
        "needs_rewrite": needs_rewrite,
    }


def node_rewrite_query(state: CorrectiveRAGState) -> CorrectiveRAGState:
    """
    NODE REWRITE QUERY: Viết lại query

    Chức năng:
    1. Analyze tại sao query không tìm được kết quả
    2. Rewrite query với từ khóa khác
    3. Tăng retry_count

    Args:
        state: State hiện tại

    Returns:
        State cập nhật với rewritten_query, retry_count++
    """
    logger.info("\n" + "="*60)
    logger.info("✍️  NODE REWRITE_QUERY: Viết lại query")
    logger.info("="*60)

    query = state["query"]
    retry_count = state.get("retry_count", 0)

    logger.info(f"📥 Query gốc: {query}")
    logger.info(f"🔄 Số lần rewrite: {retry_count}")

    # Prompt rewrite
    rewrite_prompt = f"""Bạn là trợ lý tìm kiếm thông tin tài khoản kế toán.

QUERY GỐC: {query}

Không tìm thấy kết quả phù hợp. Hãy viết lại câu hỏi theo một cách khác:
1. Sửa typo (VD: TK111 → TK 111, tt99 → TT99)
2. Dùng từ đồng nghĩa
3. Thêm context rõ hơn (VD: "tài khoản", "TK")
4. Tách query thành các query cụ thể hơn

Trả về 2-3 câu hỏi được viết lại, mỗi câu trên một dòng. Không giải thích gì thêm."""

    try:
        llm = get_ollama_client()
        response = llm.chat(
            model=settings.GENERATION_MODEL,
            messages=[{
                "role": "user",
                "content": rewrite_prompt
            }],
            options={"temperature": 0.3, "num_predict": 150},
            stream=False
        )
        rewritten = response.get("message", {}).get("content", "").strip()
        logger.info(f"✅ Query đã viết lại:\n{rewritten}")

        # Lấy query đầu tiên
        new_query = rewritten.split('\n')[0].strip()
        logger.info(f"🎯 Chọn query: {new_query}")

    except Exception as e:
        logger.warning(f"⚠️  Lỗi rewrite, giữ nguyên query: {e}")
        new_query = query

    return {
        **state,
        "rewritten_query": new_query,
        "retry_count": retry_count + 1,
    }


# =============================================================================
# BƯỚC 3-6: BUILD GRAPH
# =============================================================================

def create_coa_graph():
    """
    Xây dựng Corrective RAG workflow graph

    Cấu trúc theo idea.md:

        START → retrieve → generate_draft → grade_answer ──► END
                                      ↑                │
                                      │                ▼
                                      └───────── rewrite_query

    Loop: grade_answer → rewrite_query → retrieve (max 2 lần)
    """
    logger.info("\n🏗️  Đang xây dựng Corrective RAG Workflow Graph...")

    # BƯỚC 3: StateGraph
    graph = StateGraph(CorrectiveRAGState)

    # BƯỚC 4: add_node() cho 4 nodes
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate_draft", node_generate_draft)
    graph.add_node("grade_answer", node_grade_answer)
    graph.add_node("rewrite_query", node_rewrite_query)

    # BƯỚC 5: set_entry_point()
    graph.set_entry_point("retrieve")

    # BƯỚC 6: EDGES với conditional routing

    # retrieve → generate_draft
    graph.add_edge("retrieve", "generate_draft")

    # generate_draft → grade_answer
    graph.add_edge("generate_draft", "grade_answer")

    # grade_answer → [conditional] → END hoặc rewrite_query
    def route_after_grade(state: CorrectiveRAGState) -> Literal["end", "rewrite"]:
        """
        Conditional edge sau grade_answer

        Quyết định:
        - Nếu needs_rewrite=True và retry_count<2 → rewrite_query
        - Ngược lại → END
        """
        if state.get("needs_rewrite", False) and state.get("retry_count", 0) < 2:
            logger.info("🔀 Route: → rewrite_query")
            return "rewrite"
        logger.info("🔀 Route: → END")
        return "end"

    graph.add_conditional_edges(
        "grade_answer",
        route_after_grade,
        {
            "end": END,
            "rewrite": "rewrite_query",
        }
    )

    # rewrite_query → retrieve (loop back)
    graph.add_edge("rewrite_query", "retrieve")

    # BƯỚC 7: COMPILE với MemorySaver
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    logger.info("✅ Graph đã compile thành công\n")

    return app


# =============================================================================
# BƯỚC 8-9: RUN
# =============================================================================

def create_coa_app():
    """
    Tạo và trả về COA application

    Sử dụng:
        app = create_coa_graph()

    Returns:
        Compiled graph ready to invoke
    """
    return create_coa_graph()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test workflow
    app = create_coa_app()

    # Initial state
    initial_state: CorrectiveRAGState = {
        "messages": [],
        "query": "TK 111 là gì?",
        "rewritten_query": "",
        "documents": [],
        "answer": "",
        "confidence": 0.0,
        "retry_count": 0,
        "needs_rewrite": False,
    }

    # BƯỚC 8: INVOKE với thread_id
    config = {"configurable": {"thread_id": "test_session"}}
    result = app.invoke(initial_state, config)

    print("\n" + "█"*60)
    print("✅ HOÀN THÀNH WORKFLOW")
    print("█"*60)
    print(f"💬 Câu trả lời:\n{result['answer']}")
    print(f"📊 Confidence: {result['confidence']}")
    print(f"🔄 Retry count: {result['retry_count']}")
