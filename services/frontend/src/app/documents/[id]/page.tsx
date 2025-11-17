"use client";

import {
  ArrowLeft,
  Book,
  Calendar,
  FileText,
  LetterText,
  MessageSquare,
  Sparkles,
  Tag,
  PieChart as PieChartIcon,
} from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis } from "recharts";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAuth } from "@/contexts/AuthContext";
import { apiClient, type Document } from "@/lib/api";

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [document, setDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sentenceLimit, setSentenceLimit] = useState<number>(25);
  const [wordLimit, setWordLimit] = useState<number>(25);

  const { user } = useAuth();
  const userId = user?.id;
  const documentId = params.id as string;

  const fetchDocument = useCallback(async () => {
    if (!userId) return;

    try {
      setLoading(true);
      setError(null);
      const doc = await apiClient.getDocument(userId, documentId);
      setDocument(doc);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch document");
    } finally {
      setLoading(false);
    }
  }, [documentId, userId]);

  useEffect(() => {
    fetchDocument();
  }, [fetchDocument]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge className="bg-emerald-500/10 text-emerald-700 border-emerald-200 hover:bg-emerald-500/20">
            <span className="size-2 rounded-full bg-emerald-500 mr-2" />
            Analyzed
          </Badge>
        );
      case "processing":
        return (
          <Badge className="bg-amber-500/10 text-amber-700 border-amber-200 hover:bg-amber-500/20">
            <span className="size-2 rounded-full bg-amber-500 mr-2 animate-pulse" />
            Processing
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive">
            <span className="size-2 rounded-full bg-red-500 mr-2" />
            Failed
          </Badge>
        );
      default:
        return (
          <Badge variant="outline">
            <span className="size-2 rounded-full bg-gray-500 mr-2" />
            Unknown
          </Badge>
        );
    }
  };

  if (loading) {
    return (
      <ProtectedRoute>
        <div className="min-h-[calc(100vh-56px)] bg-background">
          <main className="mx-auto max-w-7xl px-4 py-8">
            <Skeleton className="h-10 w-32 mb-6" />
            <Skeleton className="h-12 w-full mb-8" />
            <div className="grid gap-6 md:grid-cols-2">
              <Skeleton className="h-64" />
              <Skeleton className="h-64" />
            </div>
          </main>
        </div>
      </ProtectedRoute>
    );
  }

  if (error || !document) {
    return (
      <ProtectedRoute>
        <div className="min-h-[calc(100vh-56px)] bg-background flex items-center justify-center">
          <Card className="max-w-md">
            <CardHeader>
              <CardTitle className="text-destructive">
                {error || "Document not found"}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Button
                onClick={() => router.push("/documents")}
                className="w-full"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Documents
              </Button>
            </CardContent>
          </Card>
        </div>
      </ProtectedRoute>
    );
  }

  const stats = document.stats;

  return (
    <ProtectedRoute>
      <div className="min-h-[calc(100vh-56px)] bg-background">
        <main className="mx-auto max-w-7xl px-4 py-8">
          <Button
            variant="ghost"
            onClick={() => router.push("/documents")}
            className="mb-6 -ml-2"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>

          {/* Document Header */}
          <div className="mb-8">
            <div className="flex items-start justify-between gap-4 mb-4">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <FileText className="h-6 w-6 text-muted-foreground" />
                  <h1 className="text-3xl font-bold tracking-tight">
                    {document.filename}
                  </h1>
                </div>
                <div className="flex items-center gap-4 text-muted-foreground ml-9 flex-wrap">
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    <span className="text-sm">
                      {formatDate(document.uploaded_at)}
                    </span>
                  </div>
                  {getStatusBadge(document.status)}
                  {document.status === "completed" && stats && (
                    <>
                      <Badge variant="outline" className="font-normal">
                        {stats.vocab_size.toLocaleString()} words
                      </Badge>
                      <Badge variant="outline" className="font-normal">
                        {stats.token_count.toLocaleString()} tokens
                      </Badge>
                      <Badge variant="outline" className="font-normal">
                        {(stats.type_token_ratio * 100).toFixed(2)}% lexical
                        diversity
                      </Badge>
                    </>
                  )}
                </div>
              </div>
            </div>
            <Separator />
          </div>

          {/* Error Message for Failed Documents */}
          {document.status === "failed" && document.error && (
            <Card className="mb-6 border-destructive/50 bg-destructive/5">
              <CardHeader className="p-6 pb-4">
                <CardTitle className="text-destructive">
                  Processing Failed
                </CardTitle>
                <CardDescription>
                  An error occurred during document analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                <pre className="text-sm text-destructive/80 whitespace-pre-wrap font-mono p-4 bg-destructive/10 rounded-md">
                  {document.error}
                </pre>
              </CardContent>
            </Card>
          )}

          {/* Processing Message */}
          {document.status === "processing" && (
            <Card className="mb-6 border-amber-200 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-800">
              <CardHeader className="p-6">
                <CardTitle className="text-amber-700 dark:text-amber-400">
                  Processing in Progress
                </CardTitle>
                <CardDescription className="text-amber-600 dark:text-amber-500">
                  This document is currently being analyzed. Please check back
                  in a moment.
                </CardDescription>
              </CardHeader>
            </Card>
          )}

          {/* NLP Stats - Only show if completed and stats exist */}
          {document.status === "completed" && stats && (
            <Tabs defaultValue="sentiment" className="space-y-6">
              <TabsList className="grid w-full grid-cols-2 lg:w-auto lg:grid-cols-2">
                <TabsTrigger value="sentiment">
                  <Sparkles className="mr-2 h-4 w-4" />
                  Sentiment
                </TabsTrigger>
                <TabsTrigger value="words">
                  <Book className="mr-2 h-4 w-4" />
                  Words & Grammar
                </TabsTrigger>
              </TabsList>

              {/* Words & Grammar Tab */}
              <TabsContent value="words" className="space-y-6">

                {/* Words Section */}
                <div className="space-y-6">
                  <Separator className="my-8" />
                  <div className="flex items-center gap-2">
                    <LetterText className="h-5 w-5 text-primary" />
                    <h2 className="text-xl font-semibold">Words</h2>
                  </div>

                  {/* Word Frequencies */}
                  {stats.word_frequencies &&
                    stats.word_frequencies.length > 0 && (
                      <Card>
                        <CardHeader className="p-6 pb-4">
                          <div className="flex items-center justify-between gap-4 mb-2">
                            <div className="flex items-center gap-2">
                              
                              <CardTitle>Word Frequencies</CardTitle>
                            </div>
                            <Select
                              value={wordLimit.toString()}
                              onValueChange={(value) => setWordLimit(Number(value))}
                            >
                              <SelectTrigger className="w-[140px]">
                                <SelectValue placeholder="Show" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="10">Top 10</SelectItem>
                                <SelectItem value="25">Top 25</SelectItem>
                                <SelectItem value="50">Top 50</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        </CardHeader>
                        <CardContent className="p-6">
                          <div className="h-64 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart
                                data={stats.word_frequencies
                                  .slice(0, wordLimit)
                                  .map((item: any) => ({
                                    name: item.lemma,
                                    value: item.count,
                                  }))}
                                margin={{ top: 10, right: 10, left: 0, bottom: 30 }}
                              >
                                <XAxis
                                  dataKey="name"
                                  tick={{ fontSize: 10 }}
                                  tickLine={false}
                                  axisLine={false}
                                  interval={0}
                                  angle={-45}
                                  textAnchor="end"
                                />
                                <YAxis
                                  tick={{ fontSize: 10 }}
                                  tickLine={false}
                                  axisLine={false}
                                />
                                <Tooltip
                                  cursor={{ fill: "#8b5cf633" }}
                                  content={({ active, payload }) => {
                                    if (active && payload && payload.length > 0) {
                                      const data = payload[0].payload; // access the original payload
                                      const lemma = data.name || "Unknown";
                                      const count = typeof data.value === "number" ? data.value : 0;
                                      return (
                                        <div className="bg-popover text-popover-foreground border border-border rounded-lg shadow-lg p-3 z-50">
                                          <div className="flex items-center gap-2 mb-1.5">
                                            <span className="font-semibold text-sm">{lemma}</span>
                                          </div>
                                          <div className="text-sm text-muted-foreground tabular-nums">
                                            {count}
                                          </div>
                                        </div>
                                      );
                                    }
                                    return null;
                                  }}
                                />
                                <Bar
                                  dataKey="value"
                                  fill="#8b5cf6"
                                  radius={[4, 4, 0, 0]}
                                  barSize={Math.max(8, 100 - wordLimit / 2)}
                                />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                  {/* N-grams */}
                  {stats.ngrams && (
                    <div className="grid gap-6 md:grid-cols-3">
                      {stats.ngrams.unigram &&
                        stats.ngrams.unigram.length > 0 && (
                          <Card>
                            <CardHeader className="p-6 pb-4">
                              <CardTitle className="text-base">
                                Unigrams
                              </CardTitle>
                              <CardDescription className="text-xs">
                                Most common single words
                              </CardDescription>
                            </CardHeader>
                            <CardContent className="p-6">
                              <div className="h-40 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart
                                    data={stats.ngrams.unigram
                                      .slice(0, 10)
                                      .map((item: any) => ({
                                        name: item.ngram,
                                        value: item.count,
                                      }))}
                                    margin={{ top: 5, right: 5, left: 0, bottom: 20 }}
                                  >
                                    <XAxis
                                      dataKey="name"
                                      tick={{ fontSize: 8 }}
                                      tickLine={false}
                                      axisLine={false}
                                      interval={0}
                                      angle={-30}
                                      textAnchor="end"
                                    />
                                    <YAxis
                                      tick={{ fontSize: 8 }}
                                      tickLine={false}
                                      axisLine={false}
                                    />
                                    <Tooltip
                                      cursor={{ fill: "#8b5cf633" }}
                                      content={({ active, payload }) => {
                                        if (active && payload && payload.length > 0) {
                                          const data = payload[0].payload; // access the original payload
                                          const lemma = data.name || "Unknown";
                                          const count = typeof data.value === "number" ? data.value : 0;
                                          return (
                                            <div className="bg-popover text-popover-foreground border border-border rounded-lg shadow-lg p-3 z-50">
                                              <div className="flex items-center gap-2 mb-1.5">
                                                <span className="font-semibold text-sm">{lemma}</span>
                                              </div>
                                              <div className="text-sm text-muted-foreground tabular-nums">
                                                {count}
                                              </div>
                                            </div>
                                          );
                                        }
                                        return null;
                                      }}
                                    />
                                    <Bar
                                      dataKey="value"
                                      fill="#8b5cf6"
                                      radius={[3, 3, 0, 0]}
                                      barSize={Math.max(8, 100 - 10 / 2)}
                                    />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </CardContent>
                          </Card>
                        )}

                      {stats.ngrams.bigram &&
                        stats.ngrams.bigram.length > 0 && (
                          <Card>
                            <CardHeader className="p-6 pb-4">
                              <CardTitle className="text-base">
                                Bigrams
                              </CardTitle>
                              <CardDescription className="text-xs">
                                Most common word pairs
                              </CardDescription>
                            </CardHeader>
                            <CardContent className="p-6">
                              <div className="h-40 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart
                                    data={stats.ngrams.bigram
                                      .slice(0, 10)
                                      .map((item: any) => ({
                                        name: item.ngram.replace(/_/g, " "),
                                        value: item.count,
                                      }))}
                                    margin={{ top: 5, right: 5, left: 0, bottom: 20 }}
                                  >
                                    <XAxis
                                      dataKey="name"
                                      tick={{ fontSize: 8 }}
                                      tickLine={false}
                                      axisLine={false}
                                      interval={0}
                                      angle={-30}
                                      textAnchor="end"
                                    />
                                    <YAxis
                                      tick={{ fontSize: 8 }}
                                      tickLine={false}
                                      axisLine={false}
                                    />
                                    <Tooltip
                                      cursor={{ fill: "#8b5cf633" }}
                                      content={({ active, payload }) => {
                                        if (active && payload && payload.length > 0) {
                                          const data = payload[0].payload; // access the original payload
                                          const lemma = data.name || "Unknown";
                                          const count = typeof data.value === "number" ? data.value : 0;
                                          return (
                                            <div className="bg-popover text-popover-foreground border border-border rounded-lg shadow-lg p-3 z-50">
                                              <div className="flex items-center gap-2 mb-1.5">
                                                <span className="font-semibold text-sm">{lemma}</span>
                                              </div>
                                              <div className="text-sm text-muted-foreground tabular-nums">
                                                {count}
                                              </div>
                                            </div>
                                          );
                                        }
                                        return null;
                                      }}
                                    />
                                    <Bar
                                      dataKey="value"
                                      fill="#8b5cf6"
                                      radius={[3, 3, 0, 0]}
                                      barSize={Math.max(8, 100 - 10 / 2)}
                                    />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </CardContent>
                          </Card>
                        )}

                      {stats.ngrams.trigram &&
                        stats.ngrams.trigram.length > 0 && (
                          <Card>
                            <CardHeader className="p-6 pb-4">
                              <CardTitle className="text-base">
                                Trigrams
                              </CardTitle>
                              <CardDescription className="text-xs">
                                Most common word triplets
                              </CardDescription>
                            </CardHeader>
                            <CardContent className="p-6">
                              <div className="h-40 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart
                                    data={stats.ngrams.trigram
                                      .slice(0, 10)
                                      .map((item: any) => ({
                                        name: item.ngram.replace(/_/g, " "),
                                        value: item.count,
                                      }))}
                                    margin={{ top: 5, right: 5, left: 0, bottom: 20 }}
                                  >
                                    <XAxis
                                      dataKey="name"
                                      tick={{ fontSize: 8 }}
                                      tickLine={false}
                                      axisLine={false}
                                      interval={0}
                                      angle={-30}
                                      textAnchor="end"
                                    />
                                    <YAxis
                                      tick={{ fontSize: 8 }}
                                      tickLine={false}
                                      axisLine={false}
                                    />
                                    <Tooltip
                                      cursor={{ fill: "#8b5cf633" }}
                                      content={({ active, payload }) => {
                                        if (active && payload && payload.length > 0) {
                                          const data = payload[0].payload; // access the original payload
                                          const lemma = data.name || "Unknown";
                                          const count = typeof data.value === "number" ? data.value : 0;
                                          return (
                                            <div className="bg-popover text-popover-foreground border border-border rounded-lg shadow-lg p-3 z-50">
                                              <div className="flex items-center gap-2 mb-1.5">
                                                <span className="font-semibold text-sm">{lemma}</span>
                                              </div>
                                              <div className="text-sm text-muted-foreground tabular-nums">
                                                {count}
                                              </div>
                                            </div>
                                          );
                                        }
                                        return null;
                                      }}
                                    />
                                    <Bar
                                      dataKey="value"
                                      fill="#8b5cf6"
                                      radius={[3, 3, 0, 0]}
                                      barSize={Math.max(8, 100 - 10 / 2)}
                                    />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </CardContent>
                          </Card>
                        )}
                    </div>
                  )}
                </div>

                {/* Grammar Section */}
                <div className="space-y-6">
                  <div className="flex items-center gap-2">
                    <Tag className="h-5 w-5 text-primary" />
                    <h2 className="text-xl font-semibold">Grammar</h2>
                  </div>

                  {/* POS Counts */}
                  {stats.pos_counts &&
                    Object.keys(stats.pos_counts).length > 0 && (
                      <Card>
                        <CardHeader className="p-6 pb-4">
                          <CardTitle>Part-of-Speech Tags</CardTitle>
                          <CardDescription>
                            Distribution of grammatical categories
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="p-6">
                          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                            {Object.entries(stats.pos_counts)
                              .sort(([, a], [, b]) => b - a)
                              .map(([pos, count]) => {
                                const posDescriptions: Record<string, string> =
                                  {
                                    NOUN: "Noun",
                                    VERB: "Verb",
                                    ADP: "Adposition",
                                    DET: "Determiner",
                                    PRON: "Pronoun",
                                    ADJ: "Adjective",
                                    ADV: "Adverb",
                                    CCONJ: "Coordinating Conjunction",
                                    PART: "Particle",
                                    NUM: "Numeral",
                                    X: "Other",
                                    ".": "Punctuation",
                                    PROPN: "Proper Noun",
                                    AUX: "Auxiliary Verb",
                                    SCONJ: "Subordinating Conjunction",
                                    INTJ: "Interjection",
                                    PUNCT: "Punctuation",
                                    SYM: "Symbol",
                                  };
                                const description = posDescriptions[pos] || pos;
                                return (
                                  <div
                                    key={pos}
                                    className="flex flex-col gap-1 p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                                  >
                                    <div className="flex items-center justify-between">
                                      <span className="font-medium text-sm">
                                        {description}
                                      </span>
                                      <Badge
                                        variant="secondary"
                                        className="shrink-0"
                                      >
                                        {count.toLocaleString()}
                                      </Badge>
                                    </div>
                                  </div>
                                );
                              })}
                          </div>
                        </CardContent>
                      </Card>
                    )}
                </div>
              </TabsContent>

              {/* Sentiment Tab */}
              <TabsContent value="sentiment" className="space-y-6">
                {/* Document-Level Sentiment Analysis */}
                <Card>
                  <CardHeader className="flex items-center p-6 pb-4">
                    <PieChartIcon className="h-5 w-5 text-primary" />
                    <CardTitle className="text-lg">
                      Document Sentiment
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-6">
                    {/* Helper function to get sentiment colors as hex values */}
                    {(() => {
                      const getSentimentColor = (emotionName: string): string => {
                        const normalized = emotionName.toLowerCase();
                        if (
                          normalized.includes("positive") ||
                          normalized.includes("joy") ||
                          normalized.includes("happy")
                        ) {
                          return "#EAB308"; // yellow-500
                        }
                        if (
                          normalized.includes("anger")
                        ) {
                          return "#EF4444"; // rose-500
                        }
                        if (
                          normalized.includes("negative") ||
                          normalized.includes("sad")
                        ) {
                          return "#3B82F6" ; // blue-500
                        }
                        if (normalized.includes("neutral")) {
                          return "#9CA3AF"; // gray-400
                        }
                        if (normalized.includes("fear")) {
                          return "#A855F7"; // purple-500
                        }
                        if (normalized.includes("surprise")) {
                          return "#F97316"; // orange-500
                        }
                        if (normalized.includes("disgust")) {
                          return "#10B981"; // emerald-500
                        }
                        // Default to slate for unknown emotions
                        return "#64748B"; // slate-500
                      };

                      const getSentimentLabel = (emotionName: string): string => {
                        return emotionName
                          .toLowerCase()
                          .replace(/_/g, " ")
                          .replace(/\b\w/g, (l) => l.toUpperCase());
                      };

                      // Transform sentiment data for pie chart
                      const pieData = Object.entries(stats.doc_sentiment)
                        .map(([emotion, score]) => {
                          const scoreValue = typeof score === "number" ? score : 0;
                          return {
                            name: getSentimentLabel(emotion),
                            value: scoreValue * 100,
                            color: getSentimentColor(emotion),
                            emotion: emotion,
                          };
                        })
                        .filter((item) => item.value > 0)
                        .sort((a, b) => b.value - a.value);

                      return (
                        <div className="flex flex-col md:flex-row justify-center items-center gap-8 md:gap-54">
                          <div className="w-80 h-80 flex justify-center items-center">
                            <ResponsiveContainer width="100%" height="100%">
                              <PieChart>
                                <Tooltip
                                  content={({ active, payload }) => {
                                    if (active && payload && payload.length > 0) {
                                      const data = payload[0];
                                      // Try to get color from payload, or derive it from emotion/name
                                      let color = data.payload?.color;
                                      if (!color && data.payload?.emotion) {
                                        // Fallback: derive color from emotion name
                                        const emotionName = data.payload.emotion.toLowerCase();
                                        if (emotionName.includes("positive") || emotionName.includes("joy") || emotionName.includes("happy")) {
                                          color = "#EAB308";
                                        } else if (emotionName.includes("anger")) {
                                          color = "#EF4444";
                                        } else if (emotionName.includes("negative") || emotionName.includes("sad")) {
                                          return "#3B82F6" ; 
                                        } else if (emotionName.includes("neutral")) {
                                          color = "#9CA3AF";
                                        } else if (emotionName.includes("fear")) {
                                          color = "#A855F7";
                                        } else if (emotionName.includes("surprise")) {
                                          color = "#F97316";
                                        } else if (emotionName.includes("disgust")) {
                                          color = "#10B981";
                                        } else {
                                          color = "#64748B";
                                        }
                                      }
                                      color = color || "#64748B";
                                      const name = data.name || "Unknown";
                                      const value = typeof data.value === "number" ? data.value : 0;
                                      
                                      return (
                                        <div className="bg-popover text-popover-foreground border border-border rounded-lg shadow-lg p-3 z-50">
                                          <div className="flex items-center gap-2 mb-1.5">
                                            <div
                                              className="w-3 h-3 rounded-full shrink-0"
                                              style={{
                                                backgroundColor: color,
                                              }}
                                            />
                                            <span className="font-semibold text-sm">
                                              {name}
                                            </span>
                                          </div>
                                          <div className="text-sm text-muted-foreground tabular-nums">
                                            {value.toFixed(1)}%
                                          </div>
                                        </div>
                                      );
                                    }
                                    return null;
                                  }}
                                  cursor={{ fill: "rgba(0, 0, 0, 0.05)" }}
                                />
                                <Pie
                                  data={pieData}
                                  cx="50%"
                                  cy="50%"
                                  outerRadius={120}
                                  paddingAngle={2}
                                  dataKey="value"
                                  strokeWidth={2}
                                  stroke="#fff"
                                >
                                  {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                  ))}
                                </Pie>
                              </PieChart>
                            </ResponsiveContainer>
                          </div>
                          <div className="flex flex-col gap-3">
                            <div className="flex flex-col gap-3">
                              {pieData.map((item) => (
                                <div
                                  key={item.emotion}
                                  className="flex items-center gap-3"
                                >
                                  <div
                                    className="w-4 h-4 rounded-full shrink-0"
                                    style={{ backgroundColor: item.color }}
                                  />
                                  <span className="text-sm font-medium capitalize flex-1">
                                    {item.name}
                                  </span>
                                  <span className="text-sm text-muted-foreground tabular-nums">
                                    {item.value.toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      );
                    })()}
                  </CardContent>
                </Card>

                {/* Sentence-level Sentiment Summary */}
                {stats.sentence_sentiment &&
                  stats.sentence_sentiment.length > 0 && (
                    <Card>
                      <CardHeader className="p-4 pb-3">
                        <div className="flex items-center justify-between gap-4 mb-2">
                          <div className="flex items-center gap-2">
                            <MessageSquare className="h-5 w-5 text-primary" />
                            <CardTitle>Sentence Analysis</CardTitle>
                          </div>
                          <Select
                            value={sentenceLimit.toString()}
                            onValueChange={(value) =>
                              setSentenceLimit(Number(value))
                            }
                          >
                            <SelectTrigger className="w-[140px]">
                              <SelectValue placeholder="Show" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="10">First 10</SelectItem>
                              <SelectItem value="25">First 25</SelectItem>
                              <SelectItem value="50">First 50</SelectItem>
                              <SelectItem value="100">First 100</SelectItem>
                              <SelectItem
                                value={stats.sentence_sentiment.length.toString()}
                              >
                                All ({stats.sentence_sentiment.length})
                              </SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </CardHeader>
                      <CardContent className="p-3 pt-0">
                      <div className="space-y-1.5">
                        {stats.sentence_sentiment
                          .slice(0, sentenceLimit)
                          .map((item, index) => {
                            const percentage = item.score * 100;
                            const emotionKey = item.emotion.toLowerCase().replace(/_/g, " ");

                              // Color coding for different sentiment types - modern, muted palette
                              const getSentimentColors = (emotionName: string) => {
                                const normalized = emotionName.toLowerCase();
                                if (normalized.includes("positive") || normalized.includes("joy") || normalized.includes("happy")) {
                                  return { bar: "bg-yellow-400", bg: "bg-yellow-400/20" };
                                }
                                if (normalized.includes("anger")) {
                                  return { bar: "bg-rose-400", bg: "bg-rose-400/20" };
                                }
                                if (normalized.includes("sad") || normalized.includes("negative")) {
                                  return { bar: "bg-blue-400", bg: "bg-blue-400/20" };
                                }
                                if (normalized.includes("neutral")) {
                                  return { bar: "bg-zinc-400", bg: "bg-zinc-400/20" };
                                }
                                if (normalized.includes("fear")) {
                                  return { bar: "bg-purple-400", bg: "bg-purple-400/20" };
                                }
                                if (normalized.includes("surprise")) {
                                  return { bar: "bg-orange-400", bg: "bg-orange-400/20" };
                                }
                                if (normalized.includes("disgust")) {
                                  return { bar: "bg-emerald-400", bg: "bg-emerald-400/20" };
                                }
                                // Default to slate for unknown emotions
                                return { bar: "bg-slate-400", bg: "bg-slate-400/20" };
                              };

                              const colors = getSentimentColors(item.emotion);

                              return (
                                <div
                                  key={`sentence-${index}`}
                                  className="p-2.5 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                                >
                                  <div className="flex items-center gap-1.5">
                                    <span className={`text-xs font-medium capitalize shrink-0 px-2 py-1 rounded-full text-foreground ${colors.bg}`}>
                                      {emotionKey} {percentage.toFixed(1)}%
                                    </span>
                                    <p className="text-sm text-foreground leading-relaxed line-clamp-2 flex-1">
                                      {item.sentence}
                                    </p>
                                  </div>
                                </div>
                              );
                            })}
                      </div>
                        {stats.sentence_sentiment.length > sentenceLimit && (
                          <div className="mt-4 text-sm text-muted-foreground text-center">
                            Showing top {sentenceLimit} of{" "}
                            {stats.sentence_sentiment.length} sentences
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  )}
              </TabsContent>
            </Tabs>
          )}
        </main>
      </div>
    </ProtectedRoute>
  );
}
