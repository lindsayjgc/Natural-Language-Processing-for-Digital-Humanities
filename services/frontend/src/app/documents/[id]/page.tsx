"use client";

import {
  ArrowLeft,
  Calendar,
  FileText,
  Hash,
  MessageSquare,
  Sparkles,
  Tag,
} from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
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
                  <Hash className="mr-2 h-4 w-4" />
                  Words & Grammar
                </TabsTrigger>
              </TabsList>

              {/* Words & Grammar Tab */}
              <TabsContent value="words" className="space-y-6">
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
                                    CONJ: "Conjunction",
                                    PRT: "Particle",
                                    NUM: "Numeral",
                                    X: "Other",
                                    ".": "Punctuation",
                                    PROPN: "Proper Noun",
                                    AUX: "Auxiliary Verb",
                                    SCONJ: "Subordinating Conjunction",
                                    INTJ: "Interjection",
                                    PUNCT: "Punctuation",
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
                                    <span className="text-xs text-muted-foreground font-mono">
                                      {pos}
                                    </span>
                                  </div>
                                );
                              })}
                          </div>
                        </CardContent>
                      </Card>
                    )}
                </div>

                {/* Words Section */}
                <div className="space-y-6">
                  <Separator className="my-8" />
                  <div className="flex items-center gap-2">
                    <Hash className="h-5 w-5 text-primary" />
                    <h2 className="text-xl font-semibold">Words</h2>
                  </div>

                  {/* Word Frequencies */}
                  {stats.word_frequencies &&
                    stats.word_frequencies.length > 0 && (
                      <Card>
                        <CardHeader className="p-6 pb-4">
                          <CardTitle>Word Frequencies</CardTitle>
                          <CardDescription>
                            Most frequently occurring words (top{" "}
                            {stats.word_frequencies.length})
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="p-6">
                          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                            {stats.word_frequencies.map((item, index) => (
                              <div
                                key={`${item.lemma}-${index}`}
                                className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                              >
                                <span className="font-medium text-sm truncate flex-1">
                                  {item.lemma}
                                </span>
                                <Badge
                                  variant="secondary"
                                  className="ml-2 shrink-0"
                                >
                                  {item.count}
                                </Badge>
                              </div>
                            ))}
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
                              <div className="space-y-2">
                                {stats.ngrams.unigram.map((item, index) => (
                                  <div
                                    key={`unigram-${index}`}
                                    className="flex items-center justify-between text-sm py-1.5 border-b last:border-0"
                                  >
                                    <span className="truncate flex-1">
                                      {item.ngram}
                                    </span>
                                    <Badge
                                      variant="outline"
                                      className="ml-2 shrink-0"
                                    >
                                      {item.count}
                                    </Badge>
                                  </div>
                                ))}
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
                              <div className="space-y-2">
                                {stats.ngrams.bigram.map((item, index) => (
                                  <div
                                    key={`bigram-${index}`}
                                    className="flex items-center justify-between text-sm py-1.5 border-b last:border-0"
                                  >
                                    <span className="truncate flex-1">
                                      {item.ngram.replace(/_/g, " ")}
                                    </span>
                                    <Badge
                                      variant="outline"
                                      className="ml-2 shrink-0"
                                    >
                                      {item.count}
                                    </Badge>
                                  </div>
                                ))}
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
                              <div className="space-y-2">
                                {stats.ngrams.trigram.map((item, index) => (
                                  <div
                                    key={`trigram-${index}`}
                                    className="flex items-center justify-between text-sm py-1.5 border-b last:border-0"
                                  >
                                    <span className="truncate flex-1">
                                      {item.ngram.replace(/_/g, " ")}
                                    </span>
                                    <Badge
                                      variant="outline"
                                      className="ml-2 shrink-0"
                                    >
                                      {item.count}
                                    </Badge>
                                  </div>
                                ))}
                              </div>
                            </CardContent>
                          </Card>
                        )}
                    </div>
                  )}
                </div>
              </TabsContent>

              {/* Sentiment Tab */}
              <TabsContent value="sentiment" className="space-y-6">
                {/* Document-Level Sentiment Analysis */}
                <Card>
                  <CardHeader className="p-4 pb-3">
                    <CardTitle className="text-base">
                      Document Sentiment
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-3 pt-0">
                    <div className="space-y-1.5">
                      {Object.entries(stats.doc_sentiment)
                        .sort(([, a], [, b]) => (b as number) - (a as number))
                        .map(([emotion, score]) => {
                          const scoreValue =
                            typeof score === "number" ? score : 0;
                          const percentage = scoreValue * 100;
                          const emotionKey = emotion
                            .toLowerCase()
                            .replace(/_/g, " ");

                          // Color coding for different sentiment types - modern, muted palette
                          const getSentimentColors = (emotionName: string) => {
                            const normalized = emotionName.toLowerCase();
                            if (
                              normalized.includes("positive") ||
                              normalized.includes("joy") ||
                              normalized.includes("happy")
                            ) {
                              return { bar: "bg-emerald-400", bg: "bg-emerald-400/20" };
                            }
                            if (
                              normalized.includes("negative") ||
                              normalized.includes("sad") ||
                              normalized.includes("anger")
                            ) {
                              return { bar: "bg-rose-400", bg: "bg-rose-400/20" };
                            }
                            if (normalized.includes("neutral")) {
                              return { bar: "bg-zinc-400", bg: "bg-zinc-400/20" };
                            }
                            if (normalized.includes("fear")) {
                              return { bar: "bg-violet-400", bg: "bg-violet-400/20" };
                            }
                            if (normalized.includes("surprise")) {
                              return { bar: "bg-amber-400", bg: "bg-amber-400/20" };
                            }
                            if (normalized.includes("disgust")) {
                              return { bar: "bg-orange-400", bg: "bg-orange-400/20" };
                            }
                            // Default to slate for unknown emotions
                            return { bar: "bg-slate-400", bg: "bg-slate-400/20" };
                          };

                          const colors = getSentimentColors(emotion);

                          return (
                            <div key={emotion} className="grid grid-cols-[80px_1fr_auto] gap-4 items-center">
                              <span className={`text-xs font-medium capitalize shrink-0 ${colors.bar.replace('bg-', 'text-')}`}>
                                {emotionKey}
                              </span>
                              <div
                                className={`h-3 rounded overflow-hidden border border-border/50 ${colors.bg}`}
                              >
                                <div
                                  className={`${colors.bar} h-full transition-all duration-500 ease-out`}
                                  style={{ width: `${percentage}%` }}
                                />
                              </div>
                              <span className="text-xs text-muted-foreground tabular-nums shrink-0">
                                {percentage.toFixed(1)}%
                              </span>
                            </div>
                          );
                        })}
                    </div>
                  </CardContent>
                </Card>

                {/* Sentence-level Sentiment Summary */}
                {stats.sentence_sentiment &&
                  stats.sentence_sentiment.length > 0 && (
                    <Card>
                      <CardHeader className="p-6 pb-4">
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
                              <SelectItem value="10">Top 10</SelectItem>
                              <SelectItem value="25">Top 25</SelectItem>
                              <SelectItem value="50">Top 50</SelectItem>
                              <SelectItem value="100">Top 100</SelectItem>
                              <SelectItem
                                value={stats.sentence_sentiment.length.toString()}
                              >
                                All ({stats.sentence_sentiment.length})
                              </SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </CardHeader>
                                                              <CardContent className="p-4">
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
                                  return { bar: "bg-emerald-400", bg: "bg-emerald-400/20" };
                                }
                                if (normalized.includes("negative") || normalized.includes("sad") || normalized.includes("anger")) {
                                  return { bar: "bg-rose-400", bg: "bg-rose-400/20" };
                                }
                                if (normalized.includes("neutral")) {
                                  return { bar: "bg-zinc-400", bg: "bg-zinc-400/20" };
                                }
                                if (normalized.includes("fear")) {
                                  return { bar: "bg-violet-400", bg: "bg-violet-400/20" };
                                }
                                if (normalized.includes("surprise")) {
                                  return { bar: "bg-amber-400", bg: "bg-amber-400/20" };
                                }
                                if (normalized.includes("disgust")) {
                                  return { bar: "bg-orange-400", bg: "bg-orange-400/20" };
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
