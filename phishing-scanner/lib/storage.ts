/**
 * Functions for storing and retrieving phishing detection results
 */

/**
 * Stores a URL analysis result to the backend
 * @param url The URL that was analyzed
 * @param result The analysis result containing safety status and confidence
 * @returns Promise that resolves when the storage is complete
 */
export async function storeUrlResult(url: string, result: { isSafe: boolean; safetyScore: number }): Promise<void> {
  try {
    const response = await fetch("/api/store-result", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        url,
        result,
        timestamp: new Date().toISOString(),
      }),
    })

    if (!response.ok) {
      throw new Error(`Storage API error: ${response.status}`)
    }
  } catch (error) {
    console.error("Error storing result:", error)
    throw error
  }
}

