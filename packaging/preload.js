const { contextBridge, ipcRenderer } = require("electron")

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld("electronAPI", {
  // File operations
  selectFiles: () => ipcRenderer.invoke("select-files").catch(console.error),
  selectFolder: () => ipcRenderer.invoke("select-folder").catch(console.error),
  
  // Processing operations
  processFiles: (filePaths) => ipcRenderer.invoke("process-files", filePaths).catch(console.error),
  stopProcessing: () => ipcRenderer.invoke("stop-processing").catch(console.error),
  
  // Search operations
  searchVideos: (query, options) => ipcRenderer.invoke("search-videos", query, options).catch(console.error),
  
  // System operations
  getSystemStatus: () => ipcRenderer.invoke("get-system-status").catch(console.error),
  
  // File system operations
  openFile: (filePath) => ipcRenderer.invoke("open-file", filePath).catch(console.error),
  revealFile: (filePath) => ipcRenderer.invoke("reveal-file", filePath).catch(console.error),
  generateThumbnail: (filePath, contentType) => ipcRenderer.invoke("generate-thumbnail", filePath, contentType).catch(console.error),

  // Debug operations
  getMetadataPaths: () => ipcRenderer.invoke("get-metadata-paths").catch(console.error),
  testApiConnectivity: () => ipcRenderer.invoke("test-api-connectivity").catch(console.error),

  // Event listeners with proper cleanup and error handling
  onFilesSelected: (callback) => {
    const listener = (event, filePaths) => {
      try {
        callback(filePaths)
      } catch (error) {
        console.error('Error in onFilesSelected callback:', error)
      }
    }
    ipcRenderer.on("files-selected", listener)
    return () => {
      try {
        ipcRenderer.removeListener("files-selected", listener)
      } catch (error) {
        console.error('Error removing files-selected listener:', error)
      }
    }
  },
  onFolderSelected: (callback) => {
    const listener = (event, folderPath) => {
      try {
        callback(folderPath)
      } catch (error) {
        console.error('Error in onFolderSelected callback:', error)
      }
    }
    ipcRenderer.on("folder-selected", listener)
    return () => {
      try {
        ipcRenderer.removeListener("folder-selected", listener)
      } catch (error) {
        console.error('Error removing folder-selected listener:', error)
      }
    }
  },
  onPythonOutput: (callback) => {
    const listener = (event, data) => {
      try {
        callback(data)
      } catch (error) {
        console.error('Error in onPythonOutput callback:', error)
      }
    }
    ipcRenderer.on("python-output", listener)
    return () => {
      try {
        ipcRenderer.removeListener("python-output", listener)
      } catch (error) {
        console.error('Error removing python-output listener:', error)
      }
    }
  },
  onPythonError: (callback) => {
    const listener = (event, data) => {
      try {
        callback(data)
      } catch (error) {
        console.error('Error in onPythonError callback:', error)
      }
    }
    ipcRenderer.on("python-error", listener)
    return () => {
      try {
        ipcRenderer.removeListener("python-error", listener)
      } catch (error) {
        console.error('Error removing python-error listener:', error)
      }
    }
  },
  onSearchServerOutput: (callback) => {
    const listener = (event, data) => {
      try {
        callback(data)
      } catch (error) {
        console.error('Error in onSearchServerOutput callback:', error)
      }
    }
    ipcRenderer.on("search-server-output", listener)
    return () => {
      try {
        ipcRenderer.removeListener("search-server-output", listener)
      } catch (error) {
        console.error('Error removing search-server-output listener:', error)
      }
    }
  },
  onSearchServerError: (callback) => {
    const listener = (event, data) => {
      try {
        callback(data)
      } catch (error) {
        console.error('Error in onSearchServerError callback:', error)
      }
    }
    ipcRenderer.on("search-server-error", listener)
    return () => {
      try {
        ipcRenderer.removeListener("search-server-error", listener)
      } catch (error) {
        console.error('Error removing search-server-error listener:', error)
      }
    }
  },

  // Remove listeners with error handling
  removeAllListeners: (channel) => {
    try {
      ipcRenderer.removeAllListeners(channel)
    } catch (error) {
      console.error(`Error removing all listeners for channel ${channel}:`, error)
    }
  },
})
