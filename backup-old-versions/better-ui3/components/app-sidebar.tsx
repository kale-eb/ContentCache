"use client"

import type React from "react"
import { Home, MessageSquare, BarChart3, Settings, Video, Upload } from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"

const navigationItems = [
  {
    id: "dashboard",
    title: "Dashboard",
    icon: Home,
  },
  {
    id: "upload",
    title: "Upload",
    icon: Upload,
  },
  {
    id: "ai-chat",
    title: "AI Chat",
    icon: MessageSquare,
  },
  {
    id: "analytics",
    title: "Analytics",
    icon: BarChart3,
  },
  {
    id: "settings",
    title: "Settings",
    icon: Settings,
  },
]

interface AppSidebarProps extends React.ComponentProps<typeof Sidebar> {
  activeTab: string
  onTabChange: (tab: string) => void
}

export function AppSidebar({ activeTab, onTabChange, ...props }: AppSidebarProps) {
  return (
    <Sidebar variant="inset" {...props} className="border-r border-gray-200 bg-white">
      <SidebarHeader className="border-b border-gray-100 p-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-sm">
            <Video className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-gray-900 text-lg">Silk.ai</h2>
            <p className="text-sm text-gray-500">AI-Powered Desktop</p>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent className="px-4 py-6">
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu className="space-y-2">
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.id}>
                  <SidebarMenuButton
                    onClick={() => onTabChange(item.id)}
                    isActive={activeTab === item.id}
                    className={`
                      w-full justify-start px-4 py-3 rounded-xl font-medium transition-all duration-200
                      ${
                        activeTab === item.id
                          ? "bg-blue-50 text-blue-700 border-r-4 border-blue-600 shadow-sm"
                          : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                      }
                    `}
                  >
                    <item.icon className="w-5 h-5 mr-3" />
                    <span className="text-base">{item.title}</span>
                    {item.id === "ai-chat" && (
                      <Badge className="ml-auto bg-green-100 text-green-700 text-xs">New</Badge>
                    )}
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-gray-100 p-6">
        <div className="flex items-center gap-3">
          <Avatar className="w-10 h-10">
            <AvatarImage src="/placeholder.svg?height=40&width=40" />
            <AvatarFallback className="bg-blue-100 text-blue-700 font-semibold">JD</AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-gray-900 truncate">John Doe</p>
            <p className="text-xs text-gray-500 truncate">Local User</p>
          </div>
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  )
}
