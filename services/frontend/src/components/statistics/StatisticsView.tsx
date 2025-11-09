"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { apiClient, type Document } from "@/lib/api";
import { aggregateStats, sentimentToPolarity, sentimentToAnalysis } from "@/lib/aggregateStats";
import { calculateFleschReadingEase, calculateFleschKincaidGradeLevel, formatReadabilityScore } from "@/lib/readability";
import { KeywordsCards } from "./KeywordsCards";
import { ReadabilityCards } from "./ReadabilityCards";
import { SectionBreak } from "./SectionBreak";
import { SentimentCards } from "./SentimentCards";
import { SummaryCards } from "./SummaryCards";
import { TitleSection } from "./TitleSection";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function StatisticsView() {
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [aggregatedStats, setAggregatedStats] = useState<ReturnType<typeof aggregateStats> | null>(null);

  const fetchAndAggregateStats = useCallback(async () => {
    if (!user?.id) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch all user documents
      const userDocuments = await apiClient.getUserDocuments(user.id);
      
      // Filter to only completed documents
      // Note: getUserDocuments returns documents with stats_id, but we need to fetch
      // each document individually to get the full stats
      const completedDocuments = userDocuments.documents.filter(
        (doc) => doc.status === "completed"
      );

      if (completedDocuments.length === 0) {
        setAggregatedStats(null);
        setLoading(false);
        return;
      }

      // Fetch full document details with stats for each completed document
      // Note: getUserDocuments may not include full stats, so we fetch each document
      const documentsWithStats: Array<{ stats: Document["stats"] }> = [];
      
      // Fetch documents in parallel (batch of 10 at a time to avoid overwhelming the API)
      const batchSize = 10;
      for (let i = 0; i < completedDocuments.length; i += batchSize) {
        const batch = completedDocuments.slice(i, i + batchSize);
        const batchPromises = batch.map((doc) =>
          apiClient.getDocument(user.id, doc._id).catch((err) => {
            console.error(`Failed to fetch document ${doc._id}:`, err);
            return null;
          })
        );
        const batchResults = await Promise.all(batchPromises);
        
        for (const doc of batchResults) {
          if (doc && doc.stats) {
            documentsWithStats.push({ stats: doc.stats });
          }
        }
      }

      // Aggregate stats
      const aggregated = aggregateStats(documentsWithStats);
      setAggregatedStats(aggregated);
    } catch (err) {
      console.error("Failed to fetch and aggregate stats:", err);
      setError(err instanceof Error ? err.message : "Failed to load statistics");
    } finally {
      setLoading(false);
    }
  }, [user?.id]);

  useEffect(() => {
    fetchAndAggregateStats();
  }, [fetchAndAggregateStats]);

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
        <main className="mx-auto max-w-6xl px-4 py-10">
          <TitleSection />
          <div className="space-y-8">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </div>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
        <main className="mx-auto max-w-6xl px-4 py-10">
          <TitleSection />
          <Card className="border-destructive/50 bg-destructive/5">
            <CardHeader>
              <CardTitle className="text-destructive">Error Loading Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>{error}</CardDescription>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (!aggregatedStats || aggregatedStats.totalDocuments === 0) {
    return (
      <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
        <main className="mx-auto max-w-6xl px-4 py-10">
          <TitleSection />
          <Card>
            <CardHeader>
              <CardTitle>No Statistics Available</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                You don't have any completed documents yet. Upload and process some documents to see statistics here.
              </CardDescription>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  // Calculate readability metrics
  const fleschScore = calculateFleschReadingEase(
    aggregatedStats.totalTokens,
    aggregatedStats.totalSentences
  );
  const gradeLevel = calculateFleschKincaidGradeLevel(
    aggregatedStats.totalTokens,
    aggregatedStats.totalSentences
  );

  // Transform sentiment data
  const polarityData = sentimentToPolarity(aggregatedStats.docSentiment);
  const analysisData = sentimentToAnalysis(aggregatedStats.docSentiment);

  // Prepare keyword data
  const topKeywords = aggregatedStats.wordFrequencies
    .slice(0, 5)
    .map((wf) => wf.lemma);
  
  const keywordFrequencies = aggregatedStats.wordFrequencies
    .slice(0, 10)
    .map((wf) => ({ keyword: wf.lemma, count: wf.count }));

  // Format numbers with commas
  const formatNumber = (num: number) => num.toLocaleString();

  return (
    <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
      <main className="mx-auto max-w-6xl px-4 py-10">
        <TitleSection />

        <SectionBreak title="Summary" />
        <SummaryCards
          characters={formatNumber(aggregatedStats.totalCharacters)}
          words={formatNumber(aggregatedStats.totalTokens)}
          sentences={formatNumber(aggregatedStats.totalSentences)}
        />

        <SectionBreak title="Readability" />
        <ReadabilityCards
          flesch={formatReadabilityScore(fleschScore)}
          grade={formatReadabilityScore(gradeLevel, 1)}
        />

        <SectionBreak title="Sentiment" />
        <SentimentCards
          polarityData={polarityData}
          analysisData={analysisData}
        />

        <SectionBreak title="Keywords" />
        <KeywordsCards
          topKeywords={topKeywords}
          keywordFrequencies={keywordFrequencies}
        />
      </main>
    </div>
  );
}
