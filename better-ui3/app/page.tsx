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
    setIsClient(true)
  }, [])

  if (!isClient) {
    // Show loading state during SSR to prevent hydration mismatch
    return (
      <SidebarProvider>
        <AppSidebar activeTab={activeTab} onTabChange={setActiveTab} />
        <SidebarInset className="bg-gray-50">
          <div className="flex items-center justify-center h-screen">
            <div className="text-center">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-600">Loading...</p>
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
        {/* Render all tabs but only show the active one */}
        <div style={{ display: activeTab === "dashboard" ? "block" : "none" }}>
          <DashboardTab />
        </div>
        <div style={{ display: activeTab === "upload" ? "block" : "none" }}>
          <UploadTab />
        </div>
        <div style={{ display: activeTab === "ai-chat" ? "block" : "none" }}>
          <AiChatTab />
        </div>
        <div style={{ display: activeTab === "analytics" ? "block" : "none" }}>
          <AnalyticsTab />
        </div>
        <div style={{ display: activeTab === "settings" ? "block" : "none" }}>
          <SettingsTab />
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
