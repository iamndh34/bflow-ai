┌──────────────────────────────────────────────────────────────┐
│                    BƯỚC 1: STATE DEFINITION                  │
│  CorrectiveRAGState(TypedDict)                               │
│  ├── messages: List[BaseMessage]                             │
│  ├── query: str             │ rewritten_query: str      │
│  ├── documents: List[str]   │ answer: str               │
│  ├── confidence: float      │ retry_count: int          │
│  └── needs_rewrite: bool                                 │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    BƯỚC 2: 4 NODES                           │
│  retrieve()    generate_draft()  grade_answer()  rewrite()   │
│     ↓             ↓                ↓              ↓          │
│  VectorDB      LLM Draft       Quality Score   Smart Query   │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    BƯỚC 3: StateGraph()                     │
│  workflow = StateGraph(CorrectiveRAGState)                   │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    BƯỚC 4: add_node()                       │
│  workflow.add_node("retrieve", retrieve)                     │
│  workflow.add_node("generate_draft", generate_draft)         │
│  workflow.add_node("grade", grade_answer)                    │
│  workflow.add_node("rewrite", rewrite_query)                 │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│               BƯỚC 5: set_entry_point()                     │
│                           START                              │
│                             │                                │
│                             ▼                                │
│                        [retrieve] ←─────┐                    │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    BƯỚC 6: EDGES                             │
│  START ──edge──> retrieve ──edge──> generate_draft           │
│                       │                      │               │
│                       ▼                      ▼               │
│                   [grade] ──conditional───► END              │
│                       ↑                      │               │
│                       │                      ▼               │
│                       └─────edge───── [rewrite] ───┐        │
│                                           │        │        │
└─────────────────────── LOOP (max 2) ──────────────┘        │
┌──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    BƯỚC 7: COMPILE                           │
│  checkpointer = MemorySaver()                                │
│  app = workflow.compile(checkpointer=checkpointer)           │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    BƯỚC 8: INVOKE                            │
│  config = {{"configurable": {{"thread_id": "user1"}}}}        │
│  result = app.invoke({{"query": "xuất kho"}}, config)         │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    BƯỚC 9: RESULT                            │
└──────────────────────────────────────────────────────────────┘
