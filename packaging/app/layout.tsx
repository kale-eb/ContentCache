import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Silk.AI',
  description: 'AI-Powered Content Management and Search',
  generator: 'Silk.AI',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
