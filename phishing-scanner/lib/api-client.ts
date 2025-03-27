/**
 * Client-side API functions for the phishing detection application
 */

/**
 * Sends a URL to the backend API for phishing detection analysis
 * @param url The URL to check for phishing indicators
 * @returns Analysis result with safety status and confidence score
 */
export async function checkPhishingUrl(url: string): Promise<{ isSafe: boolean; safetyScore: number }> {
  try {
    const response = await fetch("/api/analyze-url", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url }),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error("Error calling phishing detection API:", error)
    throw error
  }
}

