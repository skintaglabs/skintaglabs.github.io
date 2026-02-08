import { useState, useEffect } from 'react'
import type { AnalysisResult } from '@/types'

export interface HistoryEntry {
  id: string
  timestamp: number
  imageUrl: string
  results: AnalysisResult
  fileName: string
}

const DB_NAME = 'skintag-history'
const STORE_NAME = 'analyses'
const DB_VERSION = 1

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const objectStore = db.createObjectStore(STORE_NAME, { keyPath: 'id' })
        objectStore.createIndex('timestamp', 'timestamp', { unique: false })
      }
    }
  })
}

export function useAnalysisHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [isLoading, setIsLoading] = useState(true)

  const loadHistory = async () => {
    try {
      const db = await openDB()
      const transaction = db.transaction(STORE_NAME, 'readonly')
      const store = transaction.objectStore(STORE_NAME)
      const index = store.index('timestamp')
      const request = index.openCursor(null, 'prev')

      const entries: HistoryEntry[] = []

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result
        if (cursor) {
          entries.push(cursor.value)
          cursor.continue()
        } else {
          setHistory(entries)
          setIsLoading(false)
        }
      }

      request.onerror = () => {
        setIsLoading(false)
      }
    } catch (error) {
      console.error('Error loading history:', error)
      setIsLoading(false)
    }
  }

  const saveAnalysis = async (
    imageBlob: Blob,
    results: AnalysisResult,
    fileName: string
  ): Promise<void> => {
    try {
      const imageUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = () => reject(reader.error)
        reader.readAsDataURL(imageBlob)
      })

      const db = await openDB()
      const transaction = db.transaction(STORE_NAME, 'readwrite')
      const store = transaction.objectStore(STORE_NAME)

      const entry: HistoryEntry = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        imageUrl,
        results,
        fileName
      }

      return new Promise((resolve, reject) => {
        const request = store.add(entry)

        request.onsuccess = () => {
          loadHistory()
          resolve()
        }

        request.onerror = () => reject(request.error)
      })
    } catch (error) {
      console.error('Error saving analysis:', error)
      throw error
    }
  }

  const deleteAnalysis = async (id: string): Promise<void> => {
    try {
      const db = await openDB()
      const transaction = db.transaction(STORE_NAME, 'readwrite')
      const store = transaction.objectStore(STORE_NAME)

      return new Promise((resolve, reject) => {
        const request = store.delete(id)

        request.onsuccess = () => {
          loadHistory()
          resolve()
        }

        request.onerror = () => reject(request.error)
      })
    } catch (error) {
      console.error('Error deleting analysis:', error)
      throw error
    }
  }

  const clearHistory = async (): Promise<void> => {
    try {
      const db = await openDB()
      const transaction = db.transaction(STORE_NAME, 'readwrite')
      const store = transaction.objectStore(STORE_NAME)

      return new Promise((resolve, reject) => {
        const request = store.clear()

        request.onsuccess = () => {
          setHistory([])
          resolve()
        }

        request.onerror = () => reject(request.error)
      })
    } catch (error) {
      console.error('Error clearing history:', error)
      throw error
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  return {
    history,
    isLoading,
    saveAnalysis,
    deleteAnalysis,
    clearHistory
  }
}
