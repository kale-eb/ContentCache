// ContentCache Electron Renderer
class ContentCacheApp {
    constructor() {
        this.currentView = 'search';
        this.searchTimeout = null;
        this.currentContentType = 'all';
        
        this.init();
    }
    
    init() {
        this.setupNavigation();
        this.setupSearch();
        this.setupProcessing();
        this.setupAnalytics();
        this.setupCardActions();
        this.checkServerStatus();
        
        // Check server status periodically
        setInterval(() => this.checkServerStatus(), 10000);
    }
    
    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const views = document.querySelectorAll('.view');
        
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const viewName = item.dataset.view;
                
                // Update nav active state
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');
                
                // Update view active state
                views.forEach(v => v.classList.remove('active'));
                document.getElementById(`${viewName}View`).classList.add('active');
                
                this.currentView = viewName;
                
                // Load view-specific data
                if (viewName === 'analytics') {
                    this.loadAnalytics();
                }
            });
        });
    }
    
    setupSearch() {
        const searchInput = document.getElementById('searchInput');
        const filterBtns = document.querySelectorAll('.filter-btn');
        
        // Search input handler
        searchInput.addEventListener('input', (e) => {
            clearTimeout(this.searchTimeout);
            const query = e.target.value.trim();
            
            if (query.length > 0) {
                this.searchTimeout = setTimeout(() => {
                    this.performSearch(query);
                }, 300);
            } else {
                this.showEmptySearchState();
            }
        });
        
        // Search on Enter key
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                clearTimeout(this.searchTimeout);
                const query = e.target.value.trim();
                if (query.length > 0) {
                    this.performSearch(query);
                }
            }
        });
        
        // Filter buttons
        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentContentType = btn.dataset.type;
                
                // Re-search if there's a query
                const query = searchInput.value.trim();
                if (query.length > 0) {
                    this.performSearch(query);
                }
            });
        });
    }
    
    async performSearch(query) {
        const resultsContainer = document.getElementById('searchResults');
        
        // Show loading state
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="processing-spinner"></div>
                <h3>Searching...</h3>
                <p>Finding relevant content for "${query}"</p>
            </div>
        `;
        
        try {
            const contentTypes = this.currentContentType === 'all' ? null : [this.currentContentType];
            const results = await window.electronAPI.searchContent(query, contentTypes, 20);
            
            if (results.error) {
                this.showSearchError(results.error);
                return;
            }
            
            if (results.results && results.results.length > 0) {
                this.displaySearchResults(results.results, query);
            } else {
                this.showNoResults(query);
            }
        } catch (error) {
            console.error('Search error:', error);
            this.showSearchError(error.message);
        }
    }
    
    async displaySearchResults(results, query) {
        const resultsContainer = document.getElementById('searchResults');
        
        // Organize results by relevance buckets
        const buckets = this.organizeResultsIntoBuckets(results);
        
        // Generate cards for each result with thumbnails
        const cardPromises = results.map(result => this.createResultCard(result));
        const cards = await Promise.all(cardPromises);
        
        const bucketsHTML = Object.keys(buckets).map(bucketName => {
            const bucketResults = buckets[bucketName];
            if (bucketResults.length === 0) return '';
            
            const bucketColor = this.getBucketColor(bucketName);
            const bucketCards = bucketResults.map(result => {
                const resultIndex = results.findIndex(r => r.file_path === result.file_path);
                return cards[resultIndex];
            }).join('');
            
            return `
                <div class="result-bucket">
                    <div class="bucket-header">
                        <div class="bucket-indicator" style="background: ${bucketColor}"></div>
                        <h3>${bucketName}</h3>
                        <span class="bucket-count">${bucketResults.length}</span>
                    </div>
                    <div class="cards-grid">
                        ${bucketCards}
                    </div>
                </div>
            `;
        }).join('');
        
        resultsContainer.innerHTML = `
            <div class="search-summary">
                Found ${results.length} results for "${this.escapeHtml(query)}"
            </div>
            ${bucketsHTML}
        `;
    }
    
    async createResultCard(result) {
        const typeColor = this.getTypeColor(result.type);
        const truncatedContent = result.content ? 
            (result.content.length > 150 ? result.content.substring(0, 150) + '...' : result.content) : 
            'No content preview available';
        
        // Generate thumbnail
        let thumbnailHTML = '';
        try {
            const thumbnail = await window.electronAPI.generateThumbnail(result.file_path, result.type);
            if (thumbnail) {
                thumbnailHTML = `<img src="${thumbnail}" alt="Thumbnail" class="card-thumbnail">`;
            } else {
                thumbnailHTML = `<div class="card-thumbnail-placeholder" style="background: ${typeColor}">
                    ${this.getTypeIcon(result.type)}
                </div>`;
            }
        } catch (error) {
            console.warn('Thumbnail generation failed:', error);
            thumbnailHTML = `<div class="card-thumbnail-placeholder" style="background: ${typeColor}">
                ${this.getTypeIcon(result.type)}
            </div>`;
        }
        
        return `
            <div class="result-card" data-file-path="${this.escapeHtml(result.file_path)}">
                <div class="card-thumbnail-container">
                    ${thumbnailHTML}
                    <div class="card-overlay">
                        <div class="card-actions">
                            <button class="action-btn open-btn" title="Open File">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M9 12l2 2 4-4"/>
                                    <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/>
                                    <path d="M3 12c1 0 3 1 3 3s-2 3-3 3-3-1-3-3 2-3 3-3"/>
                                </svg>
                            </button>
                            <button class="action-btn reveal-btn" title="Show in Folder">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2l5 0 2 3h9a2 2 0 0 1 2 2z"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="relevance-score">${(result.score * 100).toFixed(0)}%</div>
                </div>
                <div class="card-content">
                    <div class="card-header">
                        <h4 class="card-title">${this.escapeHtml(result.filename || 'Unknown File')}</h4>
                        <span class="result-type" style="background: ${typeColor}">${result.type}</span>
                    </div>
                    <p class="card-description">${this.escapeHtml(truncatedContent)}</p>
                    <div class="card-path">${this.escapeHtml(this.shortenPath(result.file_path || 'Unknown path'))}</div>
                </div>
            </div>
        `;
    }
    
    organizeResultsIntoBuckets(results) {
        const buckets = {
            'Perfect Matches': [],
            'Good Matches': [],
            'Related Content': [],
            'Additional Results': []
        };
        
        results.forEach(result => {
            const score = result.score || 0;
            if (score >= 0.8) {
                buckets['Perfect Matches'].push(result);
            } else if (score >= 0.6) {
                buckets['Good Matches'].push(result);
            } else if (score >= 0.4) {
                buckets['Related Content'].push(result);
            } else {
                buckets['Additional Results'].push(result);
            }
        });
        
        return buckets;
    }
    
    getBucketColor(bucketName) {
        const colors = {
            'Perfect Matches': '#10b981',      // Green
            'Good Matches': '#3b82f6',        // Blue
            'Related Content': '#f59e0b',     // Orange
            'Additional Results': '#6b7280'   // Gray
        };
        return colors[bucketName] || '#6b7280';
    }
    
    getTypeIcon(type) {
        const icons = {
            video: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`,
            image: `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>`,
            text: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10,9 9,9 8,9"/></svg>`,
            audio: `<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/></svg>`
        };
        return icons[type] || icons.text;
    }
    
    shortenPath(filePath) {
        if (filePath.length <= 60) return filePath;
        const parts = filePath.split('/');
        if (parts.length <= 2) return filePath;
        return `.../${parts.slice(-2).join('/')}`;
    }
    
    setupCardActions() {
        // Use event delegation for dynamically created card buttons
        document.addEventListener('click', async (e) => {
            const openBtn = e.target.closest('.open-btn');
            const revealBtn = e.target.closest('.reveal-btn');
            
            if (openBtn) {
                e.preventDefault();
                const card = openBtn.closest('.result-card');
                const filePath = card?.dataset.filePath;
                
                if (filePath) {
                    try {
                        const result = await window.electronAPI.openFile(filePath);
                        if (!result.success) {
                            this.showNotification('Error', result.error || 'Failed to open file', 'error');
                        }
                    } catch (error) {
                        console.error('Failed to open file:', error);
                        this.showNotification('Error', 'Failed to open file', 'error');
                    }
                }
            }
            
            if (revealBtn) {
                e.preventDefault();
                const card = revealBtn.closest('.result-card');
                const filePath = card?.dataset.filePath;
                
                if (filePath) {
                    try {
                        const result = await window.electronAPI.revealFile(filePath);
                        if (!result.success) {
                            this.showNotification('Error', result.error || 'Failed to reveal file', 'error');
                        }
                    } catch (error) {
                        console.error('Failed to reveal file:', error);
                        this.showNotification('Error', 'Failed to reveal file', 'error');
                    }
                }
            }
        });
    }
    
    showNotification(title, message, type = 'info') {
        // Create a simple notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <strong>${title}</strong>
                <p>${message}</p>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
    
    showEmptySearchState() {
        const resultsContainer = document.getElementById('searchResults');
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"></circle>
                    <path d="m21 21-4.35-4.35"></path>
                </svg>
                <h3>Start searching</h3>
                <p>Enter a query above to search through your content</p>
            </div>
        `;
    }
    
    showNoResults(query) {
        const resultsContainer = document.getElementById('searchResults');
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"></circle>
                    <path d="m21 21-4.35-4.35"></path>
                </svg>
                <h3>No results found</h3>
                <p>No content matches "${this.escapeHtml(query)}". Try different keywords.</p>
            </div>
        `;
    }
    
    showSearchError(error) {
        const resultsContainer = document.getElementById('searchResults');
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                </svg>
                <h3>Search Error</h3>
                <p>${this.escapeHtml(error)}</p>
            </div>
        `;
    }
    
    setupProcessing() {
        const processFileCard = document.getElementById('processFileCard');
        const processDirectoryCard = document.getElementById('processDirectoryCard');
        
        processFileCard.addEventListener('click', () => this.selectAndProcessFile());
        processDirectoryCard.addEventListener('click', () => this.selectAndProcessDirectory());
        
        // Listen for processing progress
        window.electronAPI.onProcessingProgress((event, data) => {
            this.updateProcessingLog(data);
        });
    }
    
    async selectAndProcessFile() {
        try {
            const filePath = await window.electronAPI.selectFile();
            if (filePath) {
                await this.processFile(filePath);
            }
        } catch (error) {
            console.error('File selection error:', error);
            this.showProcessingError(error.message);
        }
    }
    
    async selectAndProcessDirectory() {
        try {
            const directoryPath = await window.electronAPI.selectDirectory();
            if (directoryPath) {
                await this.processDirectory(directoryPath);
            }
        } catch (error) {
            console.error('Directory selection error:', error);
            this.showProcessingError(error.message);
        }
    }
    
    async processFile(filePath) {
        this.showProcessingStatus(`Processing file: ${filePath}`);
        
        try {
            const result = await window.electronAPI.processFile(filePath);
            if (result.success) {
                this.showProcessingSuccess('File processed successfully!');
            } else {
                this.showProcessingError(result.error || 'Processing failed');
            }
        } catch (error) {
            console.error('Processing error:', error);
            this.showProcessingError(error.message || 'Processing failed');
        }
    }
    
    async processDirectory(directoryPath) {
        this.showProcessingStatus(`Processing directory: ${directoryPath}`);
        
        try {
            const result = await window.electronAPI.processDirectory(directoryPath);
            if (result.success) {
                this.showProcessingSuccess('Directory processed successfully!');
            } else {
                this.showProcessingError(result.error || 'Processing failed');
            }
        } catch (error) {
            console.error('Processing error:', error);
            this.showProcessingError(error.message || 'Processing failed');
        }
    }
    
    showProcessingStatus(message) {
        const statusDiv = document.getElementById('processingStatus');
        const logDiv = document.getElementById('processingLog');
        
        statusDiv.style.display = 'block';
        logDiv.innerHTML = `${new Date().toLocaleTimeString()}: ${message}\n`;
        
        // Scroll to processing section
        statusDiv.scrollIntoView({ behavior: 'smooth' });
    }
    
    updateProcessingLog(data) {
        const logDiv = document.getElementById('processingLog');
        if (logDiv) {
            logDiv.innerHTML += data;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
    }
    
    showProcessingSuccess(message) {
        const statusDiv = document.getElementById('processingStatus');
        const headerDiv = statusDiv.querySelector('.processing-header h3');
        const spinnerDiv = statusDiv.querySelector('.processing-spinner');
        
        headerDiv.textContent = 'Processing Complete';
        spinnerDiv.style.display = 'none';
        
        this.updateProcessingLog(`\n✅ ${message}\n`);
        
        // Hide after 3 seconds
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);
    }
    
    showProcessingError(error) {
        const statusDiv = document.getElementById('processingStatus');
        const headerDiv = statusDiv.querySelector('.processing-header h3');
        const spinnerDiv = statusDiv.querySelector('.processing-spinner');
        
        headerDiv.textContent = 'Processing Error';
        spinnerDiv.style.display = 'none';
        
        this.updateProcessingLog(`\n❌ Error: ${error}\n`);
    }
    
    async setupAnalytics() {
        // Analytics will be loaded when the view is activated
    }
    
    async loadAnalytics() {
        try {
            const status = await window.electronAPI.getSearchStatus();
            
            if (status.running && status.stats) {
                this.updateAnalyticsDisplay(status.stats);
            } else {
                this.showAnalyticsError('Search server not running or no data available');
            }
        } catch (error) {
            console.error('Analytics error:', error);
            this.showAnalyticsError(error.message);
        }
    }
    
    updateAnalyticsDisplay(stats) {
        const totalItems = (stats.video || 0) + (stats.text || 0) + (stats.image || 0) + (stats.audio || 0);
        
        document.getElementById('totalItems').textContent = totalItems.toLocaleString();
        document.getElementById('videoCount').textContent = (stats.video || 0).toLocaleString();
        document.getElementById('imageCount').textContent = (stats.image || 0).toLocaleString();
        document.getElementById('textCount').textContent = (stats.text || 0).toLocaleString();
        document.getElementById('audioCount').textContent = (stats.audio || 0).toLocaleString();
    }
    
    showAnalyticsError(error) {
        document.getElementById('totalItems').textContent = 'Error';
        document.getElementById('videoCount').textContent = '-';
        document.getElementById('imageCount').textContent = '-';
        document.getElementById('textCount').textContent = '-';
        document.getElementById('audioCount').textContent = '-';
    }
    
    async checkServerStatus() {
        const statusDiv = document.getElementById('serverStatus');
        
        try {
            const status = await window.electronAPI.getSearchStatus();
            
            if (status.running) {
                statusDiv.classList.remove('error');
                statusDiv.classList.add('connected');
                statusDiv.querySelector('span').textContent = 'Connected';
            } else {
                statusDiv.classList.remove('connected');
                statusDiv.classList.add('error');
                statusDiv.querySelector('span').textContent = 'Disconnected';
            }
        } catch (error) {
            statusDiv.classList.remove('connected');
            statusDiv.classList.add('error');
            statusDiv.querySelector('span').textContent = 'Error';
        }
    }
    
    getTypeColor(type) {
        const colors = {
            video: '#ef4444',
            image: '#10b981',
            text: '#3b82f6',
            audio: '#f59e0b'
        };
        return colors[type] || '#6b7280';
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ContentCacheApp();
}); 