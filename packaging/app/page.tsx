"use client"

import { useState, useEffect } from "react"
import { AppSidebar } from "@/components/app-sidebar"
import { DashboardTab } from "@/components/dashboard-tab"
import { UploadTab } from "@/components/upload-tab"
import { AiChatTab } from "@/components/ai-chat-tab"
import { AnalyticsTab } from "@/components/analytics-tab"
import { SettingsTab } from "@/components/settings-tab"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"

export default function Page() {
  const [activeTab, setActiveTab] = useState("dashboard")
  const [isClient, setIsClient] = useState(false)

  // Ensure this only renders on client to prevent hydration mismatches
  useEffect(() => {
    console.log("Page useEffect running, setting isClient to true")
    setIsClient(true)
  }, [])

  // Add a timeout fallback in case useEffect doesn't run
  useEffect(() => {
    const timeout = setTimeout(() => {
      console.log("Timeout fallback: forcing isClient to true")
      setIsClient(true)
    }, 1000)
    
    return () => clearTimeout(timeout)
  }, [])

  console.log("Page rendering, isClient:", isClient)

  if (!isClient) {
    // Show loading state during SSR to prevent hydration mismatch
    return (
      <SidebarProvider>
        <AppSidebar activeTab={activeTab} onTabChange={setActiveTab} />
        <SidebarInset className="bg-gray-50">
          <div className="flex items-center justify-center h-screen">
            <div className="text-center">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-600">Loading silk.ai...</p>
              <p className="text-sm text-gray-500 mt-2">Initializing application</p>
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>
    )
  }

  return (
    <SidebarProvider>
      <AppSidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <SidebarInset className="bg-gray-50">
        {/* Only render the active tab component to prevent duplicate event listeners */}
        {activeTab === "dashboard" && <DashboardTab />}
        {activeTab === "upload" && <UploadTab />}
        {activeTab === "ai-chat" && <AiChatTab />}
        {activeTab === "analytics" && <AnalyticsTab />}
        {activeTab === "settings" && <SettingsTab />}
      </SidebarInset>
    </SidebarProvider>
  )
}

