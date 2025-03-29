import { type NextRequest, NextResponse } from "next/server"

/**
 * API route handler for analyzing URLs for phishing indicators
 * This would connect to your machine learning model in a production environment
 */
export async function POST(request: NextRequest) {
  try {
    const { url } = await request.json()

    if (!url) {
      return NextResponse.json({ error: "URL is required" }, { status: 400 })
    }

    // Simulate analysis result
    const response = await fetch("https://6e81-2a09-bac1-36a0-58-00-176-7d.ngrok-free.app/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      throw new Error("Failed to get response from ML model");
    }

    const data = await response.json();

    const isSafe = data.prediction
    const safetyScore = data.score

    return NextResponse.json({
      isSafe,
      safetyScore,
    })
  } catch (error) {
    console.error("Error analyzing URL:", error)
    return NextResponse.json({ error: "Failed to analyze URL" }, { status: 500 })
  }
}



