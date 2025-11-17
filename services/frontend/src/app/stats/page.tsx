"use client";

import { BarChart3, BookOpen, Clock, Eye, TrendingUp } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/contexts/AuthContext";
import { aggregateStats, sentimentToPolarity } from "@/lib/aggregateStats";
import { apiClient, type Document } from "@/lib/api";
import {
  calculateFleschKincaidGradeLevel,
  calculateFleschReadingEase,
  formatReadabilityScore,
} from "@/lib/readability";

interface StatsData {
  totalDocuments: number;
  wordsAnalyzed: number;
  uniqueWords: number;
  sentencesAnalyzed: number;
  averageWordsPerDocument: number;
  lexicalDiversity: number;
  readabilityScore: number;
  readabilityGrade: number;
  readabilityLevel: string;
  averageSentenceLength: number;
  dominantSentiment: string;
  charactersAnalyzed: number;
  estimatedReadingTime: number;
  topKeywords: string[];
  mostCommonWord: string;
  mostCommonPhrase: string;
  sentimentBreakdown: {
    positive: number;
    neutral: number;
    negative: number;
  };
  totalBigrams: number;
  totalTrigrams: number;
}

export default function StatsPage() {
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<StatsData | null>(null);

  const fetchStats = useCallback(async () => {
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
      const completedDocuments = userDocuments.documents.filter(
        (doc) => doc.status === "completed",
      );

      if (completedDocuments.length === 0) {
        setStats({
          totalDocuments: 0,
          wordsAnalyzed: 0,
          uniqueWords: 0,
          sentencesAnalyzed: 0,
          averageWordsPerDocument: 0,
          lexicalDiversity: 0,
          readabilityScore: 0,
          readabilityGrade: 0,
          readabilityLevel: "N/A",
          averageSentenceLength: 0,
          dominantSentiment: "N/A",
          charactersAnalyzed: 0,
          estimatedReadingTime: 0,
          topKeywords: [],
          mostCommonWord: "N/A",
          mostCommonPhrase: "N/A",
          sentimentBreakdown: { positive: 0, neutral: 0, negative: 0 },
          totalBigrams: 0,
          totalTrigrams: 0,
        });
        setLoading(false);
        return;
      }

      // Fetch full document details with stats for each completed document
      const documentsWithStats: Array<{
        stats: NonNullable<Document["stats"]>;
      }> = [];

      // Fetch documents in parallel (batch of 10 at a time)
      const batchSize = 10;
      for (let i = 0; i < completedDocuments.length; i += batchSize) {
        const batch = completedDocuments.slice(i, i + batchSize);
        const batchPromises = batch.map((doc) =>
          apiClient.getDocument(user.id, doc._id).catch((err) => {
            console.error(`Failed to fetch document ${doc._id}:`, err);
            return null;
          }),
        );
        const batchResults = await Promise.all(batchPromises);

        for (const doc of batchResults) {
          if (doc?.stats) {
            documentsWithStats.push({ stats: doc.stats });
          }
        }
      }

      // Aggregate stats
      const aggregated = aggregateStats(documentsWithStats);

      // Calculate additional insights
      const averageWordsPerDocument =
        aggregated.totalDocuments > 0
          ? Math.round(aggregated.totalTokens / aggregated.totalDocuments)
          : 0;

      const lexicalDiversity = aggregated.typeTokenRatio * 100;

      const readabilityScore = calculateFleschReadingEase(
        aggregated.totalTokens,
        aggregated.totalSentences,
      );

      const readabilityGrade = calculateFleschKincaidGradeLevel(
        aggregated.totalTokens,
        aggregated.totalSentences,
      );

      // Get readability level description
      const getReadabilityLevel = (score: number): string => {
        if (score >= 90) return "Very Easy";
        if (score >= 80) return "Easy";
        if (score >= 70) return "Fairly Easy";
        if (score >= 60) return "Standard";
        if (score >= 50) return "Fairly Difficult";
        if (score >= 30) return "Difficult";
        return "Very Difficult";
      };

      const readabilityLevel = getReadabilityLevel(readabilityScore);

      const averageSentenceLength =
        aggregated.totalSentences > 0
          ? Math.round(
              (aggregated.totalTokens / aggregated.totalSentences) * 10,
            ) / 10
          : 0;

      // Estimate reading time (average 200 words per minute)
      const estimatedReadingTime = Math.ceil(aggregated.totalTokens / 200);

      // Get top keywords
      const topKeywords = aggregated.wordFrequencies
        .slice(0, 5)
        .map((wf) => wf.lemma);

      // Get most common word
      const mostCommonWord =
        aggregated.wordFrequencies.length > 0
          ? aggregated.wordFrequencies[0].lemma
          : "N/A";

      // Get most common phrase (bigram)
      const mostCommonPhrase =
        aggregated.ngrams.bigram && aggregated.ngrams.bigram.length > 0
          ? aggregated.ngrams.bigram[0].ngram.replace(/_/g, " ")
          : "N/A";

      // Find dominant sentiment
      const sentimentEntries = Object.entries(aggregated.docSentiment);
      const dominantSentiment =
        sentimentEntries.length > 0
          ? sentimentEntries
              .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))[0][0]
              .toLowerCase()
              .replace(/_/g, " ")
              .replace(/\b\w/g, (l) => l.toUpperCase())
          : "N/A";

      // Get sentiment breakdown
      const polarityData = sentimentToPolarity(aggregated.docSentiment);
      const sentimentBreakdown = {
        positive: polarityData.find((p) => p.name === "Positive")?.value || 0,
        neutral: polarityData.find((p) => p.name === "Neutral")?.value || 0,
        negative: polarityData.find((p) => p.name === "Negative")?.value || 0,
      };

      // Count n-grams
      const totalBigrams = aggregated.ngrams.bigram?.length || 0;
      const totalTrigrams = aggregated.ngrams.trigram?.length || 0;

      setStats({
        totalDocuments: aggregated.totalDocuments,
        wordsAnalyzed: aggregated.totalTokens,
        uniqueWords: aggregated.vocabSize,
        sentencesAnalyzed: aggregated.totalSentences,
        averageWordsPerDocument,
        lexicalDiversity,
        readabilityScore,
        readabilityGrade,
        readabilityLevel,
        averageSentenceLength,
        dominantSentiment,
        charactersAnalyzed: aggregated.totalCharacters,
        estimatedReadingTime,
        topKeywords,
        mostCommonWord,
        mostCommonPhrase,
        sentimentBreakdown,
        totalBigrams,
        totalTrigrams,
      });
    } catch (err) {
      console.error("Failed to fetch stats:", err);
      setError(
        err instanceof Error ? err.message : "Failed to load statistics",
      );
    } finally {
      setLoading(false);
    }
  }, [user?.id]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return (
    <ProtectedRoute>
      <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
        <main className="mx-auto max-w-7xl px-4 py-10">
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-3">
              <BarChart3 className="h-8 w-8 text-violet-600" />
              <h1 className="text-4xl font-bold">Statistics Summary</h1>
              <Badge
                variant="secondary"
                className="ml-2 bg-violet-100 text-violet-700 border-violet-200"
              >
                Overview
              </Badge>
            </div>
            <p className="text-lg text-gray-600 ml-11">
              Comprehensive analytics and insights aggregated across all your
              documents
            </p>
            <p className="text-sm text-gray-500 ml-11 mt-1">
              This is a summary view. For detailed analysis of individual
              documents, visit the document details page.
            </p>
          </div>

          {loading && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array.from({ length: 9 }, (_, i) => (
                <Skeleton
                  // biome-ignore lint/suspicious/noArrayIndexKey: Static loading skeletons, order never changes
                  key={`skeleton-loading-${i}`}
                  className="h-32 w-full"
                />
              ))}
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6">
              <p className="text-red-800">Error: {error}</p>
            </div>
          )}

          {!loading && !error && stats && (
            <>
              {/* Overview Section */}
              <div className="mb-8">
                <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                  <BookOpen className="h-6 w-6 text-violet-600" />
                  Overview Metrics
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium text-gray-600">
                        Total Documents
                      </h3>
                      <BookOpen className="h-4 w-4 text-gray-400" />
                    </div>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.totalDocuments.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Completed documents
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium text-gray-600">
                        Words Analyzed
                      </h3>
                      <TrendingUp className="h-4 w-4 text-gray-400" />
                    </div>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.wordsAnalyzed.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Total tokens processed
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium text-gray-600">
                        Unique Words
                      </h3>
                      <Eye className="h-4 w-4 text-gray-400" />
                    </div>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.uniqueWords.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Vocabulary size
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium text-gray-600">
                        Reading Time
                      </h3>
                      <Clock className="h-4 w-4 text-gray-400" />
                    </div>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.estimatedReadingTime}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {stats.estimatedReadingTime === 1 ? "minute" : "minutes"}{" "}
                      @ 200 WPM
                    </p>
                  </div>
                </div>
              </div>

              {/* Text Analysis Section */}
              <div className="mb-8">
                <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="h-6 w-6 text-violet-600" />
                  Text Analysis
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Sentences Analyzed
                    </h3>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.sentencesAnalyzed.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Total sentences
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Avg Words/Document
                    </h3>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.averageWordsPerDocument.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Average document length
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Avg Sentence Length
                    </h3>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.averageSentenceLength.toFixed(1)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Words per sentence
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Lexical Diversity
                    </h3>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.lexicalDiversity.toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Type-token ratio
                    </p>
                  </div>
                </div>
              </div>

              {/* Readability & Complexity Section */}
              <div className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">
                  Readability & Complexity
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-violet-50 to-purple-50 p-6 rounded-xl border border-violet-200 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Readability Score
                    </h3>
                    <p className="text-4xl font-bold text-violet-600 mb-1">
                      {formatReadabilityScore(stats.readabilityScore)}
                    </p>
                    <Badge className="bg-violet-100 text-violet-700 border-violet-200 mt-2">
                      {stats.readabilityLevel}
                    </Badge>
                    <p className="text-xs text-gray-600 mt-2">
                      Flesch Reading Ease (0-100)
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Grade Level
                    </h3>
                    <p className="text-4xl font-bold text-violet-600">
                      {stats.readabilityGrade.toFixed(1)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Flesch-Kincaid Grade Level
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      U.S. school grade equivalent
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Characters Analyzed
                    </h3>
                    <p className="text-3xl font-bold text-violet-600">
                      {stats.charactersAnalyzed.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Total characters (estimated)
                    </p>
                  </div>
                </div>
              </div>

              {/* Sentiment Analysis Section */}
              <div className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">
                  Sentiment Analysis
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Dominant Sentiment
                    </h3>
                    <p className="text-2xl font-bold text-violet-600 capitalize mb-2">
                      {stats.dominantSentiment}
                    </p>
                    <p className="text-xs text-gray-500">Most common emotion</p>
                  </div>

                  <div className="bg-green-50 p-6 rounded-xl border border-green-200 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-green-700">
                      Positive
                    </h3>
                    <p className="text-3xl font-bold text-green-600">
                      {stats.sentimentBreakdown.positive.toFixed(1)}%
                    </p>
                    <div className="mt-2 w-full bg-green-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full"
                        style={{
                          width: `${stats.sentimentBreakdown.positive}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="bg-gray-50 p-6 rounded-xl border border-gray-200 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-gray-700">
                      Neutral
                    </h3>
                    <p className="text-3xl font-bold text-gray-600">
                      {stats.sentimentBreakdown.neutral.toFixed(1)}%
                    </p>
                    <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-gray-600 h-2 rounded-full"
                        style={{
                          width: `${stats.sentimentBreakdown.neutral}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="bg-red-50 p-6 rounded-xl border border-red-200 shadow-sm">
                    <h3 className="text-sm font-medium mb-2 text-red-700">
                      Negative
                    </h3>
                    <p className="text-3xl font-bold text-red-600">
                      {stats.sentimentBreakdown.negative.toFixed(1)}%
                    </p>
                    <div className="mt-2 w-full bg-red-200 rounded-full h-2">
                      <div
                        className="bg-red-600 h-2 rounded-full"
                        style={{
                          width: `${stats.sentimentBreakdown.negative}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Language Patterns Section */}
              <div className="mb-8">
                <h2 className="text-2xl font-semibold mb-4">
                  Language Patterns
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-3 text-gray-700">
                      Most Common Word
                    </h3>
                    <p className="text-2xl font-bold text-violet-600 capitalize mb-2">
                      {stats.mostCommonWord}
                    </p>
                    <p className="text-xs text-gray-500">
                      Across all documents
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-3 text-gray-700">
                      Most Common Phrase
                    </h3>
                    <p className="text-lg font-bold text-violet-600 capitalize mb-2 line-clamp-2">
                      "{stats.mostCommonPhrase}"
                    </p>
                    <p className="text-xs text-gray-500">
                      Most frequent bigram
                    </p>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                    <h3 className="text-sm font-medium mb-3 text-gray-700">
                      Top Keywords
                    </h3>
                    <div className="flex flex-wrap gap-2 mb-2">
                      {stats.topKeywords.length > 0 ? (
                        stats.topKeywords.map((keyword) => (
                          <Badge
                            key={keyword}
                            variant="secondary"
                            className="bg-violet-100 text-violet-700 border-violet-200"
                          >
                            {keyword}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-sm text-gray-500">N/A</span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500">
                      Top 5 most frequent words
                    </p>
                  </div>
                </div>
              </div>

              {/* N-gram Statistics */}
              {stats.totalBigrams > 0 || stats.totalTrigrams > 0 ? (
                <div className="mb-8">
                  <h2 className="text-2xl font-semibold mb-4">
                    N-gram Statistics
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                      <h3 className="text-sm font-medium mb-2 text-gray-700">
                        Unique Bigrams
                      </h3>
                      <p className="text-3xl font-bold text-violet-600">
                        {stats.totalBigrams.toLocaleString()}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Unique two-word phrases
                      </p>
                    </div>

                    <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
                      <h3 className="text-sm font-medium mb-2 text-gray-700">
                        Unique Trigrams
                      </h3>
                      <p className="text-3xl font-bold text-violet-600">
                        {stats.totalTrigrams.toLocaleString()}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Unique three-word phrases
                      </p>
                    </div>
                  </div>
                </div>
              ) : null}
            </>
          )}

          {!loading && !error && stats && stats.totalDocuments === 0 && (
            <div className="bg-white p-6 rounded-xl border border-black/10 shadow-sm">
              <p className="text-gray-600">
                You don't have any completed documents yet. Upload and process
                some documents to see statistics here.
              </p>
            </div>
          )}
        </main>
      </div>
    </ProtectedRoute>
  );
}
