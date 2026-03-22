/**
 * WebSocket client for ATP Dashboard real-time updates.
 *
 * Provides automatic reconnection, message handling, and subscription management.
 *
 * @example
 * const ws = new ATPWebSocket();
 *
 * ws.on('test_progress', (data) => {
 *     console.log('Test progress:', data);
 * });
 *
 * ws.subscribe('test:progress', { suite_execution_id: 123 });
 *
 * ws.connect();
 */

class ATPWebSocket {
    /**
     * Create a new ATPWebSocket client.
     *
     * @param {Object} options - Configuration options
     * @param {string} options.url - WebSocket URL (auto-detected if not provided)
     * @param {boolean} options.autoReconnect - Enable automatic reconnection (default: true)
     * @param {number} options.reconnectInterval - Base reconnection interval in ms (default: 1000)
     * @param {number} options.maxReconnectInterval - Maximum reconnection interval in ms (default: 30000)
     * @param {number} options.reconnectDecay - Reconnection interval decay factor (default: 1.5)
     * @param {number} options.maxReconnectAttempts - Max reconnection attempts (default: null = infinite)
     * @param {number} options.pingInterval - Ping interval in ms (default: 25000)
     */
    constructor(options = {}) {
        this.options = {
            url: options.url || this._getDefaultUrl(),
            autoReconnect: options.autoReconnect !== false,
            reconnectInterval: options.reconnectInterval || 1000,
            maxReconnectInterval: options.maxReconnectInterval || 30000,
            reconnectDecay: options.reconnectDecay || 1.5,
            maxReconnectAttempts: options.maxReconnectAttempts || null,
            pingInterval: options.pingInterval || 25000,
        };

        this.ws = null;
        this.clientId = null;
        this.isConnected = false;
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.reconnectTimeout = null;
        this.pingIntervalId = null;

        // Event handlers
        this._handlers = new Map();

        // Pending subscriptions (for reconnection)
        this._pendingSubscriptions = new Map();

        // Last sequence number for delta updates
        this._lastSequence = 0;

        // Bind methods
        this._onOpen = this._onOpen.bind(this);
        this._onClose = this._onClose.bind(this);
        this._onError = this._onError.bind(this);
        this._onMessage = this._onMessage.bind(this);
    }

    /**
     * Get the default WebSocket URL based on current page location.
     *
     * @returns {string} WebSocket URL
     */
    _getDefaultUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/api/ws/updates`;
    }

    /**
     * Connect to the WebSocket server.
     *
     * @returns {Promise<void>} Resolves when connected
     */
    connect() {
        return new Promise((resolve, reject) => {
            if (this.isConnected || this.isConnecting) {
                resolve();
                return;
            }

            this.isConnecting = true;

            // Build URL with client ID for reconnection
            let url = this.options.url;
            if (this.clientId) {
                url += `?client_id=${encodeURIComponent(this.clientId)}`;
            }

            try {
                this.ws = new WebSocket(url);

                this.ws.onopen = (event) => {
                    this._onOpen(event);
                    resolve();
                };

                this.ws.onclose = this._onClose;
                this.ws.onerror = (event) => {
                    this._onError(event);
                    if (!this.isConnected) {
                        reject(new Error('WebSocket connection failed'));
                    }
                };
                this.ws.onmessage = this._onMessage;

            } catch (error) {
                this.isConnecting = false;
                reject(error);
            }
        });
    }

    /**
     * Disconnect from the WebSocket server.
     */
    disconnect() {
        this.options.autoReconnect = false;
        this._clearReconnectTimeout();
        this._clearPingInterval();

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.isConnected = false;
        this.isConnecting = false;
    }

    /**
     * Subscribe to a topic.
     *
     * @param {string} topic - Topic to subscribe to
     * @param {Object} filter - Optional filter for the subscription
     */
    subscribe(topic, filter = {}) {
        // Store for reconnection
        this._pendingSubscriptions.set(topic, filter);

        if (this.isConnected) {
            this._send({
                type: 'subscribe',
                payload: { topic, filter }
            });
        }
    }

    /**
     * Unsubscribe from a topic.
     *
     * @param {string} topic - Topic to unsubscribe from
     */
    unsubscribe(topic) {
        this._pendingSubscriptions.delete(topic);

        if (this.isConnected) {
            this._send({
                type: 'unsubscribe',
                payload: { topic }
            });
        }
    }

    /**
     * Register an event handler.
     *
     * @param {string} eventType - Event type to handle
     * @param {Function} handler - Handler function
     */
    on(eventType, handler) {
        if (!this._handlers.has(eventType)) {
            this._handlers.set(eventType, new Set());
        }
        this._handlers.get(eventType).add(handler);
    }

    /**
     * Remove an event handler.
     *
     * @param {string} eventType - Event type
     * @param {Function} handler - Handler function to remove
     */
    off(eventType, handler) {
        const handlers = this._handlers.get(eventType);
        if (handlers) {
            handlers.delete(handler);
        }
    }

    /**
     * Remove all handlers for an event type.
     *
     * @param {string} eventType - Event type
     */
    offAll(eventType) {
        this._handlers.delete(eventType);
    }

    /**
     * Get the last received sequence number for delta updates.
     *
     * @returns {number} Last sequence number
     */
    getLastSequence() {
        return this._lastSequence;
    }

    // Private methods

    _onOpen(event) {
        this.isConnected = true;
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        console.log('[ATPWebSocket] Connected');

        // Start ping interval
        this._startPingInterval();

        // Emit connect event
        this._emit('connect', { event });
    }

    _onClose(event) {
        this.isConnected = false;
        this.isConnecting = false;
        this._clearPingInterval();

        console.log('[ATPWebSocket] Disconnected', event.code, event.reason);

        // Emit disconnect event
        this._emit('disconnect', { event, code: event.code, reason: event.reason });

        // Attempt reconnection
        if (this.options.autoReconnect) {
            this._scheduleReconnect();
        }
    }

    _onError(event) {
        console.error('[ATPWebSocket] Error:', event);
        this._emit('error', { event });
    }

    _onMessage(event) {
        try {
            const message = JSON.parse(event.data);
            this._handleMessage(message);
        } catch (error) {
            console.error('[ATPWebSocket] Failed to parse message:', error);
        }
    }

    _handleMessage(message) {
        const { type, payload, sequence, timestamp } = message;

        // Track sequence for delta updates
        if (sequence !== undefined && sequence !== null) {
            if (sequence > this._lastSequence + 1 && this._lastSequence > 0) {
                console.warn('[ATPWebSocket] Missed sequences:', this._lastSequence + 1, 'to', sequence - 1);
                this._emit('sequence_gap', {
                    expected: this._lastSequence + 1,
                    received: sequence
                });
            }
            this._lastSequence = sequence;
        }

        // Handle system messages
        switch (type) {
            case 'connected':
                this.clientId = payload.client_id;
                console.log('[ATPWebSocket] Assigned client ID:', this.clientId);
                // Resubscribe to pending topics
                this._resubscribe();
                break;

            case 'subscribed':
                console.log('[ATPWebSocket] Subscribed to:', payload.topic);
                break;

            case 'unsubscribed':
                console.log('[ATPWebSocket] Unsubscribed from:', payload.topic);
                break;

            case 'pong':
                // Ping response received
                break;

            case 'error':
                console.error('[ATPWebSocket] Server error:', payload.error);
                break;
        }

        // Emit to handlers
        this._emit(type, payload, { timestamp, sequence });
    }

    _emit(eventType, data, meta = {}) {
        const handlers = this._handlers.get(eventType);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(data, meta);
                } catch (error) {
                    console.error('[ATPWebSocket] Handler error:', error);
                }
            });
        }

        // Also emit to wildcard handlers
        const wildcardHandlers = this._handlers.get('*');
        if (wildcardHandlers) {
            wildcardHandlers.forEach(handler => {
                try {
                    handler(eventType, data, meta);
                } catch (error) {
                    console.error('[ATPWebSocket] Wildcard handler error:', error);
                }
            });
        }
    }

    _send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    _scheduleReconnect() {
        if (this.options.maxReconnectAttempts !== null &&
            this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.log('[ATPWebSocket] Max reconnect attempts reached');
            this._emit('reconnect_failed', { attempts: this.reconnectAttempts });
            return;
        }

        this._clearReconnectTimeout();

        // Calculate delay with exponential backoff
        const delay = Math.min(
            this.options.reconnectInterval * Math.pow(this.options.reconnectDecay, this.reconnectAttempts),
            this.options.maxReconnectInterval
        );

        console.log(`[ATPWebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);

        this.reconnectTimeout = setTimeout(async () => {
            this.reconnectAttempts++;
            this._emit('reconnecting', { attempt: this.reconnectAttempts });

            try {
                await this.connect();
            } catch (error) {
                console.error('[ATPWebSocket] Reconnection failed:', error);
            }
        }, delay);
    }

    _resubscribe() {
        // Resubscribe to all pending topics after reconnection
        for (const [topic, filter] of this._pendingSubscriptions) {
            this._send({
                type: 'subscribe',
                payload: { topic, filter }
            });
        }
    }

    _startPingInterval() {
        this._clearPingInterval();
        this.pingIntervalId = setInterval(() => {
            if (this.isConnected) {
                this._send({ type: 'ping' });
            }
        }, this.options.pingInterval);
    }

    _clearPingInterval() {
        if (this.pingIntervalId) {
            clearInterval(this.pingIntervalId);
            this.pingIntervalId = null;
        }
    }

    _clearReconnectTimeout() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
    }
}

/**
 * React hook for using ATPWebSocket in React components.
 *
 * @param {Object} options - WebSocket options
 * @returns {Object} WebSocket state and methods
 *
 * @example
 * function TestProgress() {
 *     const { isConnected, subscribe, on } = useATPWebSocket();
 *
 *     useEffect(() => {
 *         on('test_progress', (data) => {
 *             setProgress(data.progress_percent);
 *         });
 *         subscribe('test:progress', { suite_execution_id: 123 });
 *     }, [on, subscribe]);
 *
 *     return <div>Connected: {isConnected ? 'Yes' : 'No'}</div>;
 * }
 */
function useATPWebSocket(options = {}) {
    const wsRef = React.useRef(null);
    const [isConnected, setIsConnected] = React.useState(false);
    const [lastMessage, setLastMessage] = React.useState(null);

    React.useEffect(() => {
        const ws = new ATPWebSocket(options);
        wsRef.current = ws;

        ws.on('connect', () => setIsConnected(true));
        ws.on('disconnect', () => setIsConnected(false));
        ws.on('*', (type, data, meta) => setLastMessage({ type, data, meta }));

        ws.connect().catch(console.error);

        return () => {
            ws.disconnect();
        };
    }, []);

    const subscribe = React.useCallback((topic, filter) => {
        wsRef.current?.subscribe(topic, filter);
    }, []);

    const unsubscribe = React.useCallback((topic) => {
        wsRef.current?.unsubscribe(topic);
    }, []);

    const on = React.useCallback((eventType, handler) => {
        wsRef.current?.on(eventType, handler);
        return () => wsRef.current?.off(eventType, handler);
    }, []);

    return {
        isConnected,
        lastMessage,
        subscribe,
        unsubscribe,
        on,
        ws: wsRef.current,
    };
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.ATPWebSocket = ATPWebSocket;
    window.useATPWebSocket = useATPWebSocket;
}
