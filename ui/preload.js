const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // File/Directory selection
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  selectFile: () => ipcRenderer.invoke('select-file'),
  
  // Processing operations
  processFile: (filePath) => ipcRenderer.invoke('process-file', filePath),
  processDirectory: (directoryPath) => ipcRenderer.invoke('process-directory', directoryPath),
  
  // Search operations
  searchContent: (query, contentTypes, topK) => ipcRenderer.invoke('search-content', query, contentTypes, topK),
  getSearchStatus: () => ipcRenderer.invoke('get-search-status'),
  
  // Progress updates
  onProcessingProgress: (callback) => ipcRenderer.on('processing-progress', callback),
  removeProcessingProgressListener: (callback) => ipcRenderer.removeListener('processing-progress', callback)
}); 