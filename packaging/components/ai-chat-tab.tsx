"use client"

import { useState, useEffect } from "react"
import { Send, Bot, User, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"

interface Message {
  id: number
  type: "user" | "ai"
  content: string
  timestamp: Date
}

export function AiChatTab() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      type: "ai",
      content:
        "Hello! I'm your AI assistant for video content analysis. I can help you search through your videos, analyze content, generate summaries, and answer questions about your video library. What would you like to know?",
      timestamp: new Date(),
    },
  ])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (typeof window !== "undefined" && window.electronAPI) {
      window.electronAPI.onPythonOutput((event, data) => {
        try {
          const response = JSON.parse(data)

          if (response.type === "ai_response") {
            const aiMessage: Message = {
              id: Date.now(),
              type: "ai",
              content: response.message,
              timestamp: new Date(),
            }
            setMessages((prev) => [...prev, aiMessage])
            setIsLoading(false)
          } else if (response.type === "search_results") {
            // Handle search results in AI context
            let searchSummary = `I found ${response.results.length} results:\n\n`
            response.results.slice(0, 5).forEach((result: any, index: number) => {
              searchSummary += `${index + 1}. ${result.filename || result.title || "Unknown file"}\n`
            })

            const aiMessage: Message = {
              id: Date.now(),
              type: "ai",
              content: searchSummary,
              timestamp: new Date(),
            }
            setMessages((prev) => [...prev, aiMessage])
            setIsLoading(false)
          }
        } catch (error) {
          // Handle non-JSON output
          console.log("Non-JSON Python output:", data)
        }
      })

      return () => {
        window.electronAPI.removeAllListeners("python-output")
      }
    }
  }, [])

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return

    const newMessage: Message = {
      id: Date.now(),
      type: "user",
      content: inputMessage,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, newMessage])
    setInputMessage("")
    setIsLoading(true)

    // Send to Python AI via Electron
    if (window.electronAPI) {
      await window.electronAPI.aiChat(inputMessage)
    }
  }

  const quickPrompts = [
    "Summarize my recent videos",
    "Find videos about product demos",
    "What topics are covered in my library?",
    "Show me the longest videos",
  ]

  return (
    <div className="flex-1 flex flex-col h-full p-8">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-blue-600 rounded-xl flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">AI Chat</h1>
            <p className="text-gray-600">Intelligent video content analysis and search</p>
          </div>
          <Badge className="ml-auto bg-green-100 text-green-700">Local AI</Badge>
        </div>
      </div>

      {/* Chat Container */}
      <Card className="flex-1 flex flex-col border-0 shadow-sm bg-white">
        {/* Messages */}
        <CardContent className="flex-1 p-6 overflow-y-auto">
          <div className="space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-4 ${message.type === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.type === "ai" && (
                  <Avatar className="w-8 h-8 bg-gradient-to-br from-purple-600 to-blue-600">
                    <AvatarFallback className="bg-transparent">
                      <Bot className="w-4 h-4 text-white" />
                    </AvatarFallback>
                  </Avatar>
                )}
                <div
                  className={`max-w-[70%] p-4 rounded-2xl ${
                    message.type === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900"
                  }`}
                >
                  <p className="text-sm leading-relaxed">{message.content}</p>
                  <p className={`text-xs mt-2 ${message.type === "user" ? "text-blue-100" : "text-gray-500"}`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
                {message.type === "user" && (
                  <Avatar className="w-8 h-8 bg-blue-600">
                    <AvatarFallback className="bg-transparent">
                      <User className="w-4 h-4 text-white" />
                    </AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-4 justify-start">
                <Avatar className="w-8 h-8 bg-gradient-to-br from-purple-600 to-blue-600">
                  <AvatarFallback className="bg-transparent">
                    <Bot className="w-4 h-4 text-white" />
                  </AvatarFallback>
                </Avatar>
                <div className="bg-gray-100 text-gray-900 p-4 rounded-2xl">
                  <p className="text-sm">Thinking...</p>
                </div>
              </div>
            )}
          </div>
        </CardContent>

        {/* Quick Prompts */}
        <div className="px-6 py-4 border-t border-gray-100">
          <p className="text-sm text-gray-600 mb-3">Quick prompts:</p>
          <div className="flex flex-wrap gap-2">
            {quickPrompts.map((prompt, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                className="text-xs"
                onClick={() => setInputMessage(prompt)}
              >
                {prompt}
              </Button>
            ))}
          </div>
        </div>

        {/* Input */}
        <div className="p-6 border-t border-gray-100">
          <div className="flex gap-3">
            <Input
              placeholder="Ask me anything about your video content..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
              className="flex-1 h-12 text-base border-gray-200 focus:border-blue-500 focus:ring-blue-500"
              disabled={isLoading}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="h-12 px-6 bg-blue-600 hover:bg-blue-700"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}
