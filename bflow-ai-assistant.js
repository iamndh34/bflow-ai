/**
 * BFlow AI Chatbot - Accounting Assistant
 * Professional UI with streaming support
 */
$(document).ready(function () {

    const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    const API_BASE = isLocal
        ? "http://localhost:8010/api/ai-bflow"  // Local
        : "/api/ai-bflow";                      // Production (Relative path)
    const API_ASK = `${API_BASE}/posting-engine/ask`;
    const API_HISTORY = `${API_BASE}/posting-engine/get_history`;
    const API_RESET = `${API_BASE}/posting-engine/reset_history`;
    const API_SESSIONS = `${API_BASE}/sessions`;
    const API_SESSION_CREATE = `${API_BASE}/sessions/create`;
    const API_SESSION_DELETE = `${API_BASE}/sessions/delete`;
    const API_SESSION_DETAIL = `${API_BASE}/sessions`; // + /{session_id}

    // DOM Elements
    const $chatBtn     = $('#aiChatButton');
    const $chatBox     = $('#aiChatContainer');
    const $chatClose   = $('#aiChatClose');
    const $chatNew     = $('#aiChatNew');
    const $chatSessions = $('#aiChatSessions');
    const $chatBody    = $('#aiChatBody');
    const $input       = $('#aiChatInput');
    const $sendBtn     = $('#aiChatSend');
    const $chatType    = $('#aiChatType');

    // Sessions offcanvas elements
    const $sessionsOffcanvas = $('#aiSessionsOffcanvas');
    const $sessionsBackdrop  = $('#aiSessionsBackdrop');
    const $sessionsClose     = $('#aiSessionsClose');
    const $sessionsList      = $('#aiSessionsList');

    let isStreaming = false;
    let currentSessionId = null;

    // Track session per mode
    let modeSessions = {
        thinking: { sessionId: null, messages: '' },
        free: { sessionId: null, messages: '' }
    };

    // Sample questions data (loaded from JSON file)
    let sampleQuestions = {
        financial: { posting_engine: [], coa: [], compare: [] },
        default: { general: [] }
    };

    /**
     * Load sample questions from JSON file
     */
    function loadSampleQuestions() {
        const jsonPath = $chatBox.data('sample-questions-url');
        if (!jsonPath) {
            console.warn('[AI Chat] Sample questions URL not found');
            return;
        }
        $.getJSON(jsonPath, function(data) {
            sampleQuestions = data;
        }).fail(function() {
            console.warn('[AI Chat] Failed to load sample questions from JSON file');
        });
    }

    // Load sample questions on page load
    loadSampleQuestions();

    // Loading messages that rotate
    const loadingMessages = [
        "Đã nhận câu hỏi...",
        "Đang phân tích yêu cầu...",
        "Đang suy nghĩ...",
        "Bạn đợi một chút nhé...",
    ];

    /**
     * Scroll to bottom of chat
     */
    function scrollBottom() {
        $chatBody.scrollTop($chatBody[0].scrollHeight);
    }

    /**
     * Format timestamp
     */
    function formatTime(date) {
        const d = date ? new Date(date) : new Date();
        return d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
    }

    /**
     * Add message to chat
     */
    function addMessage(content, sender, timestamp) {
        const safe = sender === 'ai'
            ? DOMPurify.sanitize(marked.parse(content))
            : $('<div>').text(content).html();
        const time = formatTime(timestamp);
        const currentMode = $chatType.val();
        const modeClass = sender === 'user' ? ` mode-${currentMode}` : '';

        const $msg = $(`
            <div class="ai-message ${sender}">
                <div class="ai-message-content${modeClass}">${safe}</div>
                <div class="ai-message-time">${time}</div>
            </div>
        `);
        $chatBody.append($msg);
        scrollBottom();
    }

    /**
     * Show suggestions chips based on current space
     */
    function showSuggestions() {
        const currentSpace = $('#menu-tenant').attr('data-space-selected');
        const isFinancial = currentSpace === 'financial';

        let suggestionsHtml = '';

        if (isFinancial) {
            // Financial space: hiển thị câu hỏi từ sampleQuestions.financial
            const questions = sampleQuestions.financial || {};
            const postingHtml = (questions.posting_engine || []).map(q =>
                `<button class="ai-suggestion-chip ai-gradient-5" data-question="${q}">${q}</button>`
            ).join('');
            const coaHtml = (questions.coa || []).map(q =>
                `<button class="ai-suggestion-chip ai-gradient-1" data-question="${q}">${q}</button>`
            ).join('');
            const compareHtml = (questions.compare || []).map(q =>
                `<button class="ai-suggestion-chip ai-gradient-6" data-question="${q}">${q}</button>`
            ).join('');
            suggestionsHtml = postingHtml + coaHtml + compareHtml;
        } else {
            // Other spaces: hiển thị câu hỏi từ sampleQuestions.default
            const questions = sampleQuestions.default || {};
            const generalHtml = (questions.general || []).map(q =>
                `<button class="ai-suggestion-chip ai-gradient-1" data-question="${q}">${q}</button>`
            ).join('');
            suggestionsHtml = generalHtml;
        }

        // Nếu không có câu hỏi nào, không hiển thị suggestions
        if (!suggestionsHtml) {
            return;
        }

        const $suggestionsWrapper = $(`
            <div class="ai-suggestions">
                ${suggestionsHtml}
            </div>
        `);
        $chatBody.append($suggestionsWrapper);

        // Handle suggestion clicks
        $suggestionsWrapper.find('.ai-suggestion-chip').on('click', function() {
            const question = $(this).data('question');
            $input.val(question);
            $sendBtn.prop('disabled', false);
            sendMessage();
        });

        scrollBottom();
    }

    /**
     * Show welcome message and suggestions based on chat type
     */
    function showWelcomeMessage() {
        const chatType = $chatType.val();

        if (chatType === 'free') {
            // Free mode: simple welcome message only
            const welcomeText = `Xin chào, tôi có thể giúp gì cho bạn?`;
            addMessage(welcomeText, 'ai');
        } else {
            // Thinking mode: full welcome message with suggestions
            const welcomeText = `Xin chào! Tôi là trợ lý AI của BFlow.\n\nTôi có thể giúp bạn giải đáp các thắc mắc về nghiệp vụ và các quy trình hệ thống của BFlow.`;
            addMessage(welcomeText, 'ai');
            showSuggestions();
        }
    }

    /**
     * Create AI bubble with loading text
     */
    function createStreamingBubble() {
        const $msg = $(`
            <div class="ai-message ai">
                <div class="ai-message-content">
                    <div class="ai-streaming-wrapper">
                        <span class="ai-loading-text">${loadingMessages[0]}</span>
                    </div>
                    <div class="ai-final" style="display:none"></div>
                </div>
                <div class="ai-message-time" style="display:none"></div>
            </div>
        `);
        $chatBody.append($msg);
        scrollBottom();

        // Rotate loading messages (10s each, no repeat)
        let msgIndex = 0;
        const $loadingText = $msg.find('.ai-loading-text');
        const loadingInterval = setInterval(() => {
            msgIndex++;
            if (msgIndex >= loadingMessages.length) {
                clearInterval(loadingInterval);
                return;
            }
            $loadingText.fadeOut(200, function() {
                $(this).text(loadingMessages[msgIndex]).fadeIn(200);
            });
        }, 5000);

        return {
            $wrapper: $msg.find('.ai-streaming-wrapper'),
            $final: $msg.find('.ai-final'),
            $time: $msg.find('.ai-message-time'),
            stopLoading: () => clearInterval(loadingInterval)
        };
    }

    /**
     * Send message to AI
     */
    async function sendMessage() {
        const question = $input.val().trim();
        const chatType = $chatType.val();
        console.log('[AI Chat] sendMessage called:', { question, chatType, isStreaming });

        if (!question || isStreaming) {
            console.log('[AI Chat] sendMessage blocked:', { hasQuestion: !!question, isStreaming });
            return;
        }

        // Add user message
        addMessage(question, 'user');
        $input.val('').css('height', 'auto');
        $sendBtn.prop('disabled', true);

        isStreaming = true;

        // Create session if not exists
        if (!currentSessionId) {
            try {
                console.log('[AI Chat] Creating new session for:', chatType);
                const createRes = await fetch(`${API_SESSION_CREATE}?chat_type=${chatType}`, { method: 'POST' });
                if (createRes.ok) {
                    const createData = await createRes.json();
                    currentSessionId = createData.session_id;
                    console.log('[AI Chat] Session created via API:', currentSessionId);
                }
            } catch (err) {
                console.error('[AI Chat] Failed to create session:', err);
            }
        }

        // Create streaming bubble with loading text
        const { $wrapper, $final, $time, stopLoading } = createStreamingBubble();

        try {
            const chatType = $chatType.val();
            let url = `${API_ASK}?question=${encodeURIComponent(question)}&chat_type=${chatType}`;
            if (currentSessionId) {
                url += `&session_id=${currentSessionId}`;
            }
            console.log('[AI Chat] Sending message:', { url, chatType, currentSessionId });
            const res = await fetch(url);
            console.log('[AI Chat] Response received:', res.status);

            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');

            let buffer = '';
            let firstToken = true;
            let sessionParsed = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;

                // Parse and remove session_id from first line if present
                if (!sessionParsed && buffer.includes('\n')) {
                    const firstLine = buffer.split('\n')[0];
                    if (firstLine.startsWith('__SESSION_ID__:')) {
                        currentSessionId = firstLine.replace('__SESSION_ID__:', '').trim();
                        buffer = buffer.substring(buffer.indexOf('\n') + 1);
                    }
                    sessionParsed = true;
                }

                // Get actual content (after session parsing)
                const content = sessionParsed ? buffer : '';

                // First real content received - switch from loading to final
                if (firstToken && content.length > 0) {
                    stopLoading();
                    $wrapper.hide();
                    $final.show();
                    firstToken = false;
                }

                // Render markdown realtime
                if (!firstToken && content.length > 0) {
                    const html = DOMPurify.sanitize(marked.parse(content));
                    $final.html(html);
                    scrollBottom();
                }
            }

            // After streaming ends, ensure content is displayed
            const finalContent = buffer;
            if (finalContent && firstToken) {
                stopLoading();
                $wrapper.hide();
                $final.show();
                const html = DOMPurify.sanitize(marked.parse(finalContent));
                $final.html(html);
                scrollBottom();
            }

            // Show time after streaming complete
            $time.text(formatTime()).show();
        } catch (err) {
            // Stop loading and show error
            stopLoading();
            $wrapper.html(`<span class="ai-error"><span class="ai-error-icon">!</span> Hiện tại tôi không thể kết nối tới máy chủ. Vui lòng liên hệ admin để được giải quyết nhé.</span>`);
            console.error('AI Chat Error:', err);
        } finally {
            isStreaming = false;
            $sendBtn.prop('disabled', !$input.val().trim());
        }
    }

    /**
     * Show loading indicator
     */
    function showLoading(text = 'Đang tải lịch sử...') {
        const $loading = $(`
            <div class="ai-loading-history">
                <div class="ai-loading-spinner"></div>
                <span>${text}</span>
            </div>
        `);
        $chatBody.append($loading);
    }

    /**
     * Hide loading indicator
     */
    function hideLoading() {
        $('.ai-loading-history').remove();
    }

    /**
     * Load conversation history from server
     */
    async function loadHistory() {
        showLoading();

        try {
            const chatType = $chatType.val();
            let url = `${API_HISTORY}?chat_type=${chatType}`;
            if (currentSessionId) {
                url += `&session_id=${currentSessionId}`;
            }
            const res = await fetch(url);
            if (!res.ok) {
                hideLoading();
                return false;
            }

            const data = await res.json();
            if (!data.history || data.history.length === 0) {
                hideLoading();
                return false;
            }

            hideLoading();

            // Show note about history limit
            const $note = $(`
                <div class="ai-history-note">
                    Hiển thị 10 câu hỏi gần nhất
                </div>
            `);
            $chatBody.append($note);

            // Display history
            data.history.forEach(item => {
                // User question
                addMessage(item.question, 'user', item.time);

                // AI response
                if (item.response) {
                    addMessage(item.response, 'ai', item.time);
                }
            });

            // Show suggestions after history (only for thinking mode)
            if (chatType === 'thinking') {
                showSuggestions();
            }

            return true;
        } catch (err) {
            console.error('Failed to load context:', err);
            hideLoading();
            return false;
        }
    }

    /**
     * Clear chat history and reset server context
     */
    async function clearChat() {
        $chatBody.empty();
        showLoading('Đang xoá lịch sử...');

        try {
            const chatType = $chatType.val();
            let url = `${API_RESET}?chat_type=${chatType}`;
            if (currentSessionId) {
                url += `&session_id=${currentSessionId}`;
            }
            await fetch(url, { method: 'POST' });
        } catch (err) {
            console.error('Failed to reset context:', err);
        }

        hideLoading();
        showWelcomeMessage();
    }

    /* =========================================
       Sessions Management
       ========================================= */

    /**
     * Format date for session display
     */
    function formatSessionDate(dateStr) {
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Vừa xong';
        if (diffMins < 60) return `${diffMins} phút trước`;
        if (diffHours < 24) return `${diffHours} giờ trước`;
        if (diffDays < 7) return `${diffDays} ngày trước`;
        return date.toLocaleDateString('vi-VN');
    }

    /**
     * Open sessions offcanvas
     */
    function openSessionsPanel() {
        $sessionsOffcanvas.addClass('active');
        $sessionsBackdrop.addClass('active');
        loadSessions();
    }

    /**
     * Close sessions offcanvas
     */
    function closeSessionsPanel() {
        $sessionsOffcanvas.removeClass('active');
        $sessionsBackdrop.removeClass('active');
    }

    /**
     * Load sessions list from server
     */
    async function loadSessions() {
        $sessionsList.html(`
            <div class="ai-sessions-empty">
                <div class="ai-loading-spinner"></div>
                <span>Đang tải...</span>
            </div>
        `);

        try {
            // Fetch sessions from both modes and merge
            console.log('[AI Chat] Loading sessions from both modes...');

            const [thinkingRes, freeRes] = await Promise.all([
                fetch(`${API_SESSIONS}?chat_type=thinking`),
                fetch(`${API_SESSIONS}?chat_type=free`)
            ]);

            let allSessions = [];

            if (thinkingRes.ok) {
                const thinkingData = await thinkingRes.json();
                allSessions = allSessions.concat(thinkingData.sessions || []);
            }

            if (freeRes.ok) {
                const freeData = await freeRes.json();
                allSessions = allSessions.concat(freeData.sessions || []);
            }

            // Sort by updated_at descending (newest first)
            allSessions.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));

            console.log('[AI Chat] All sessions:', allSessions);
            renderSessions(allSessions);
        } catch (err) {
            console.error('[AI Chat] Failed to load sessions:', err);
            $sessionsList.html(`
                <div class="ai-sessions-empty">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 8v4M12 16h.01"/>
                    </svg>
                    <span>Không thể tải lịch sử</span>
                </div>
            `);
        }
    }

    /**
     * Render sessions list
     */
    function renderSessions(sessions) {
        if (!sessions.length) {
            $sessionsList.html(`
                <div class="ai-sessions-empty">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                    </svg>
                    <span>Chưa có cuộc trò chuyện nào</span>
                </div>
            `);
            return;
        }

        const html = sessions.map(session => {
            const isActive = session.id === currentSessionId;
            const modeClass = session.chat_type === 'free' ? 'mode-free' : 'mode-thinking';

            return `
                <div class="ai-session-item ${isActive ? 'active' : ''} ${modeClass}" data-session-id="${session.id}" data-session-mode="${session.chat_type}">
                    <div class="ai-session-info">
                        <div class="ai-session-title">${$('<div>').text(session.title || 'Cuộc trò chuyện mới').html()}</div>
                        <div class="ai-session-meta">
                            <span>${formatSessionDate(session.updated_at)}</span>
                        </div>
                    </div>
                    <button class="ai-session-delete" data-session-id="${session.id}" data-session-mode="${session.chat_type}" title="Xóa">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                        </svg>
                    </button>
                </div>
            `;
        }).join('');

        $sessionsList.html(html);

        // Bind click events
        $sessionsList.find('.ai-session-item').on('click', function(e) {
            if ($(e.target).closest('.ai-session-delete').length) return;
            const sessionId = $(this).data('session-id');
            const sessionMode = $(this).data('session-mode');
            loadSession(sessionId, sessionMode);
        });

        $sessionsList.find('.ai-session-delete').on('click', function(e) {
            e.stopPropagation();
            const sessionId = $(this).data('session-id');
            const sessionMode = $(this).data('session-mode');
            deleteSession(sessionId, sessionMode);
        });
    }

    /**
     * Load a specific session and switch to its mode
     */
    async function loadSession(sessionId, sessionMode) {
        closeSessionsPanel();
        $chatBody.empty();
        showLoading();

        try {
            const res = await fetch(`${API_SESSION_DETAIL}/${sessionId}`);
            if (!res.ok) throw new Error('Failed to load session');

            const data = await res.json();
            currentSessionId = sessionId;

            // Switch to session's mode
            const mode = sessionMode || data.chat_type || 'thinking';
            switchMode(mode);

            hideLoading();

            if (!data.history || data.history.length === 0) {
                showWelcomeMessage();
                return;
            }

            // Display history
            data.history.forEach(item => {
                addMessage(item.question, 'user', item.time);
                if (item.response) {
                    addMessage(item.response, 'ai', item.time);
                }
            });

            // Show suggestions for thinking mode
            if (mode === 'thinking') {
                showSuggestions();
            }

            // Save to modeSessions
            modeSessions[mode].sessionId = sessionId;
            modeSessions[mode].messages = $chatBody.html();
        } catch (err) {
            console.error('Failed to load session:', err);
            hideLoading();
            showWelcomeMessage();
        }
    }

    /**
     * Switch chat mode (UI only, no API call)
     */
    function switchMode(mode) {
        // Update active state
        $('.ai-mode-btn').removeClass('active');
        $(`.ai-mode-btn[data-mode="${mode}"]`).addClass('active');

        // Update hidden input value
        $chatType.val(mode);

        // Update title color
        $('.ai-chat-title').removeClass('mode-thinking mode-free').addClass('mode-' + mode);

        // Update send button color
        $sendBtn.removeClass('mode-thinking mode-free').addClass('mode-' + mode);

        // Update container mode class
        $chatBox.removeClass('mode-thinking mode-free').addClass('mode-' + mode);

        // Update existing user messages color
        $('.ai-message.user .ai-message-content').removeClass('mode-thinking mode-free').addClass('mode-' + mode);
    }

    /**
     * Delete a session
     */
    async function deleteSession(sessionId, sessionMode) {
        if (!confirm('Xóa cuộc trò chuyện này?')) return;

        try {
            const chatType = sessionMode || $chatType.val();
            await fetch(`${API_SESSION_DELETE}?chat_type=${chatType}&session_id=${sessionId}`, {
                method: 'DELETE'
            });

            // If deleting current session, start new chat
            if (sessionId === currentSessionId) {
                currentSessionId = null;
                $chatBody.empty();
                showWelcomeMessage();
            }

            // Reload sessions list
            loadSessions();
        } catch (err) {
            console.error('Failed to delete session:', err);
        }
    }

    /**
     * Start a new chat session (just clear UI, session will be created on first message)
     */
    function startNewChat() {
        const currentMode = $chatType.val();

        // Reset current mode's saved state
        currentSessionId = null;
        modeSessions[currentMode].sessionId = null;
        modeSessions[currentMode].messages = '';

        $chatBody.empty();
        showWelcomeMessage();
    }

    /* =========================================
       Event Handlers
       ========================================= */

    // Toggle chat window
    $chatBtn.on('click', () => {
        $chatBox.addClass('active');
        if ($chatBody.children().length === 0) {
            showWelcomeMessage();
        }
        $input.focus();
    });

    // Close chat
    $chatClose.on('click', () => {
        $chatBox.removeClass('active');
    });

    // New chat button
    $chatNew.on('click', () => {
        startNewChat();
    });

    // Open sessions panel
    $chatSessions.on('click', () => {
        openSessionsPanel();
    });

    // Close sessions panel
    $sessionsClose.on('click', closeSessionsPanel);
    $sessionsBackdrop.on('click', closeSessionsPanel);

    // Mode switch buttons - switch to mode's saved conversation
    $('.ai-mode-btn').on('click', function() {
        const $btn = $(this);
        if ($btn.hasClass('active')) return;

        const currentMode = $chatType.val();
        const newMode = $btn.data('mode');

        // Save current mode's state
        modeSessions[currentMode].sessionId = currentSessionId;
        modeSessions[currentMode].messages = $chatBody.html();

        // Switch UI to new mode
        switchMode(newMode);

        // Restore new mode's state
        currentSessionId = modeSessions[newMode].sessionId;
        const savedMessages = modeSessions[newMode].messages;
        if (savedMessages && savedMessages.trim()) {
            $chatBody.html(savedMessages);
            scrollBottom();
        } else {
            $chatBody.empty();
            showWelcomeMessage();
        }
    });

    // Auto-resize textarea
    $input.on('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        $sendBtn.prop('disabled', !this.value.trim() || isStreaming);
    });

    // Send on Enter (Shift+Enter for newline)
    $input.on('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Send button click
    $sendBtn.on('click', sendMessage);

    // Close on Escape
    $(document).on('keydown', e => {
        if (e.key === 'Escape' && $chatBox.hasClass('active')) {
            $chatBox.removeClass('active');
        }
    });

    // Initialize mode classes on page load
    (function initModeClasses() {
        const initialMode = $chatType.val() || 'thinking';
        $sendBtn.addClass('mode-' + initialMode);
        $('.ai-chat-title').addClass('mode-' + initialMode);
        $chatBox.addClass('mode-' + initialMode);
    })();
});
