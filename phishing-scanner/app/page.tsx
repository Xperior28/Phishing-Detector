"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { AlertCircle, CheckCircle, Shield } from "lucide-react"
import { checkPhishingUrl } from "@/lib/api-client"
import { storeUrlResult } from "@/lib/storage"

export default function Home() {
  const [url, setUrl] = useState("")
  const [result, setResult] = useState<null | { isSafe: boolean; safetyScore: number }>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  const handleCheckUrl = async () => {
    if (!url) return

    setIsLoading(true)
    try {
      const data = await checkPhishingUrl(url)
      console.log(data.isSafe)
      console.log(data.safetyScore)
      setResult(data)
    } catch (error) {
      console.error("Error checking URL:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleStoreResult = async () => {
    if (!result) return

    setIsSaving(true)
    try {
      await storeUrlResult(url, result)
      alert("Result saved successfully")
    } catch (error) {
      console.error("Error storing result:", error)
      alert("Failed to save result")
    } finally {
      setIsSaving(false)
    }
  }

  const clearResults = () => {
    setResult(null)
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-background text-foreground relative">
      <div className="max-w-4xl w-full text-center space-y-2 mb-12 relative z-10">
        <h1 className="text-3xl font-bold tracking-tight">DEPARTMENT OF INFORMATION TECHNOLOGY</h1>
        <h2 className="text-2xl font-semibold tracking-tight">NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA, SURATHKAL</h2>
        <h3 className="text-xl font-medium mt-4 text-muted-foreground">
          Information Assurance and Security (IT352) Course Project
        </h3>
        <h4 className="text-2xl font-bold text-primary mt-2">Phishing URL Detection Using Machine Learning</h4>

        <div className="mt-6 text-muted-foreground">
          <p>Carried out by</p>
          <p className="font-medium">Rohith V (221IT055)</p>
          <p className="font-medium">Vaibhav (221IT076)</p>
          <p className="italic mt-2">During Academic Session January - April 2025</p>
        </div>
      </div>

      <Card className="w-full max-w-2xl p-6 shadow-lg border-primary/20 backdrop-blur-sm bg-card/80 relative z-10">
        <div className="space-y-6">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-gradient-to-r from-green-500 to-primary rounded-full p-2 mr-3">
              <Shield className="h-6 w-6 text-background" />
            </div>
            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-primary">
              PhishGuard
            </h2>
          </div>

          <div className="space-y-2">
            <label htmlFor="url-input" className="text-sm font-medium">
              Enter URL to analyze for phishing threats
            </label>
            <div className="flex gap-2">
              <Input
                id="url-input"
                placeholder="example.com"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="flex-1"
              />
              <Button
                onClick={handleCheckUrl}
                disabled={!url || isLoading}
                className="whitespace-nowrap bg-green-600 hover:bg-green-700 text-white"
              >
                {isLoading ? "Analyzing..." : "Analyze URL"}
              </Button>
            </div>
          </div>

          {result && (
            <div
              className={`border rounded-lg p-4 ${result.isSafe ? "bg-green-950/20 border-green-800" : "bg-red-950/20 border-red-800"}`}
            >
              <div className="flex items-center gap-2 mb-2">
                <h3 className="font-medium text-lg">Analysis Result:</h3>
                {result.isSafe ? (
                  <div className="flex items-center text-green-500">
                    <CheckCircle className="h-5 w-5 mr-1" />
                    Safe URL
                  </div>
                ) : (
                  <div className="flex items-center text-red-500">
                    <AlertCircle className="h-5 w-5 mr-1" />
                    Potential Phishing URL
                  </div>
                )}
              </div>
              <p>Safety Score: {result.safetyScore.toFixed(2)}%</p>
              <p className="text-sm text-muted-foreground mt-2">URL: {url}</p>
            </div>
          )}

          {result && (
            <div className="flex flex-col sm:flex-row gap-3 pt-2">
              <Button
                variant="outline"
                onClick={clearResults}
                disabled={isLoading}
                className="flex-1 border-green-700/50 hover:bg-green-950/20 hover:text-green-400"
              >
                Clear Results
              </Button>
              <Button
                onClick={handleStoreResult}
                disabled={!result || isLoading || isSaving}
                className="flex-1 bg-green-600 hover:bg-green-700 text-white"
              >
                {isSaving ? "Saving..." : "Save Results"}
              </Button>
            </div>
          )}
        </div>
      </Card>
    </main>
  )
}

