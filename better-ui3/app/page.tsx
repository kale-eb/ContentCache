"use client"

import { useState } from "react"
import { AppSidebar } from "@/components/app-sidebar"
import { DashboardTab } from "@/components/dashboard-tab"
import { UploadTab } from "@/components/upload-tab"
import { AiChatTab } from "@/components/ai-chat-tab"
import { AnalyticsTab } from "@/components/analytics-tab"
import { SettingsTab } from "@/components/settings-tab"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"

export default function Page() {
  const [activeTab, setActiveTab] = useState("dashboard")

  const renderActiveTab = () => {
    switch (activeTab) {
      case "dashboard":
        return <DashboardTab />
      case "upload":
        return <UploadTab />
      case "ai-chat":
        return <AiChatTab />
      case "analytics":
        return <AnalyticsTab />
      case "settings":
        return <SettingsTab />
      default:
        return <DashboardTab />
    }
  }

  return (
    <SidebarProvider>
      <AppSidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <SidebarInset className="bg-gray-50">{renderActiveTab()}</SidebarInset>
    </SidebarProvider>
  )
}
