import { type NextRequest, NextResponse } from "next/server"
import * as fs from 'fs';
import * as path from 'path';

/**
 * API route handler for storing URL analysis results
 * In a production environment, this would store to a database
 */
export async function POST(request: NextRequest) {
  try {
    const data = await request.json()
    const { url, result, timestamp } = data
    const FILE_PATH = 'results.csv';

    if (!url || !result) {
      return NextResponse.json({ error: "URL and result are required" }, { status: 400 })
    }

    // In a real implementation, you would store this in a database
    // For demonstration, we'll log it and simulate storage
    console.log("Storing result:", { url, result, timestamp })

    const newRow = `"${url}",${result.isSafe},${result.safetyScore},"${timestamp}\n"`;

    const header = 'URL,IsSafe,SafetyScore,Timestamp\n';
    if (!fs.existsSync(FILE_PATH)) {
        fs.writeFileSync(FILE_PATH, header);
    }
    fs.appendFileSync(FILE_PATH, newRow);

    // Simulate storage operation
    await new Promise((resolve) => setTimeout(resolve, 500))

    return NextResponse.json({
      success: true,
      message: "Result stored successfully",
    })
  } catch (error) {
    console.error("Error storing result:", error)
    return NextResponse.json({ error: "Failed to store result" }, { status: 500 })
  }
}

