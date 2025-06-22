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
    
    displaySearchResults(results, query) {
        const resultsContainer = document.getElementById('searchResults');
        
        const resultsHTML = results.map(result => {
            const typeColor = this.getTypeColor(result.type);
            const truncatedContent = result.content ? 
                (result.content.length > 200 ? result.content.substring(0, 200) + '...' : result.content) : 
                'No content preview available';
            
            return `
                <div class="search-result">
                    <div class="result-header">
                        <div>
                            <div class="result-title">${this.escapeHtml(result.filename || 'Unknown File')}</div>
                            <span class="result-type" style="background: ${typeColor}">${result.type}</span>
                        </div>
                        <div class="result-score">${(result.score * 100).toFixed(1)}%</div>
                    </div>
                    <div class="result-content">${this.escapeHtml(truncatedContent)}</div>
                    <div class="result-path">${this.escapeHtml(result.file_path || 'Unknown path')}</div>
                </div>
            `;
        }).join('');
        
        resultsContainer.innerHTML = `
            <div style="margin-bottom: 20px; color: #999; font-size: 14px;">
                Found ${results.length} results for "${this.escapeHtml(query)}"
            </div>
            ${resultsHTML}
        `;
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