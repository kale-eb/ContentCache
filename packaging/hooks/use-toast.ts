"use client"

// Inspired by react-hot-toast library
import * as React from "react"
import { useState } from "react"

interface Toast {
  title: string
  description?: string
  variant?: "default" | "destructive"
}

interface ToasterToast extends Toast {
  id: string
  open: boolean
}

const TOAST_LIMIT = 1
const TOAST_REMOVE_DELAY = 3000

let count = 0

function genId() {
  count = (count + 1) % Number.MAX_SAFE_INTEGER
  return count.toString()
}

const addToRemoveQueue = (toastId: string) => {
  const timeout = setTimeout(() => {
    dispatch({
      type: "REMOVE_TOAST",
      toastId: toastId,
    })
  }, TOAST_REMOVE_DELAY)

  return timeout
}

type ActionType = "ADD_TOAST" | "UPDATE_TOAST" | "DISMISS_TOAST" | "REMOVE_TOAST"

type Action = {
  type: ActionType
  toast?: ToasterToast
}

interface State {
  toasts: ToasterToast[]
}

const toastTimeouts = new Map<string, ReturnType<typeof setTimeout>>()

const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case "ADD_TOAST":
      return {
        ...state,
        toasts: [action.toast!, ...state.toasts].slice(0, TOAST_LIMIT),
      }

    case "UPDATE_TOAST":
      return {
        ...state,
        toasts: state.toasts.map((t) => (t.id === action.toast!.id ? { ...t, ...action.toast } : t)),
      }

    case "DISMISS_TOAST": {
      const { toast } = action

      if (toast) {
        toastTimeouts.set(toast.id, addToRemoveQueue(toast.id))
      } else {
        state.toasts.forEach((toast) => {
          toastTimeouts.set(toast.id, addToRemoveQueue(toast.id))
        })
      }

      return {
        ...state,
        toasts: state.toasts.map((t) =>
          t.id === toast?.id || toast === undefined
            ? {
                ...t,
                open: false,
              }
            : t,
        ),
      }
    }
    case "REMOVE_TOAST":
      if (action.toast === undefined) {
        return {
          ...state,
          toasts: [],
        }
      }
      return {
        ...state,
        toasts: state.toasts.filter((t) => t.id !== action.toast.id),
      }
  }
}

const listeners: Array<(state: State) => void> = []

let memoryState: State = { toasts: [] }

function dispatch(action: Action) {
  memoryState = reducer(memoryState, action)
  listeners.forEach((listener) => {
    listener(memoryState)
  })
}

function useToast() {
  const [toasts, setToasts] = useState<ToasterToast[]>(memoryState.toasts)

  React.useEffect(() => {
    listeners.push(setToasts)
    return () => {
      const index = listeners.indexOf(setToasts)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }, [toasts])

  const toast = (toast: Toast) => {
    const id = genId()
    const newToast: ToasterToast = { ...toast, id, open: true }

    dispatch({
      type: "ADD_TOAST",
      toast: newToast,
    })

    // Auto-remove toast after 3 seconds
    setTimeout(() => {
      dispatch({ type: "REMOVE_TOAST", toast: newToast })
    }, TOAST_REMOVE_DELAY)
  }

  return { toast, toasts }
}

export { useToast }
