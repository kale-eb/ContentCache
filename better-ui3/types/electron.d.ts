export interface ElectronAPI {
  // File operations
  selectFiles: () => Promise<string[]>
  selectFolder: () => Promise<string>
  
  // Processing operations
  processFiles: (filePaths: string[]) => Promise<{ success: boolean }>
  
  // Search operations
  searchVideos: (query: string, options?: {
    content_types?: string[]
    top_k?: number
  }) => Promise<{ success: boolean }>
  
  // System operations
  getSystemStatus: () => Promise<{ success: boolean }>
  
  // Event listeners with cleanup
  onFilesSelected: (callback: (filePaths: string[]) => void) => () => void
  onFolderSelected: (callback: (folderPath: string) => void) => () => void
  onPythonOutput: (callback: (output: string) => void) => () => void
  onPythonError: (callback: (error: string) => void) => () => void
  onSearchServerOutput: (callback: (output: string) => void) => () => void
  onSearchServerError: (callback: (error: string) => void) => () => void
  
  // File system operations
  openFile: (filePath: string) => Promise<void>
  revealFile: (filePath: string) => Promise<void>
  generateThumbnail: (filePath: string, contentType: string) => Promise<string>
  
  // Utility
  removeAllListeners: (channel: string) => void
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export {} 