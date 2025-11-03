"use client";

import { ArrowLeft, Calendar } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { apiClient, type Document } from "@/lib/api";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [document, setDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Mock user ID - in a real app this would come from auth
  // const userId = "demo_user";
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
          <Badge
            variant="default"
            className="bg-emerald-100 text-emerald-700 border-emerald-200"
          >
            <span className="size-2 rounded-full bg-emerald-500 mr-2" />
            Analyzed
          </Badge>
        );
      case "processing":
        return (
          <Badge
            variant="secondary"
            className="bg-amber-100 text-amber-700 border-amber-200"
          >
            <span className="size-2 rounded-full bg-amber-500 mr-2" />
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
          <Badge variant="outline" className="text-gray-600 border-gray-300">
            <span className="size-2 rounded-full bg-gray-500 mr-2" />
            Unknown
          </Badge>
        );
    }
  };

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-56px)] bg-gray-50 flex items-center justify-center">
        <div className="text-gray-600">Loading document...</div>
      </div>
    );
  }

  if (error || !document) {
    return (
      <div className="min-h-[calc(100vh-56px)] bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-600 mb-4">
            {error || "Document not found"}
          </div>
          <Button onClick={() => router.push("/documents")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Documents
          </Button>
        </div>
      </div>
    );
  }

  const stats = document.stats;

  return (
    <ProtectedRoute>
      <div className="min-h-[calc(100vh-56px)] bg-gray-50">
        <main className="mx-auto max-w-6xl px-4 py-10">
          <Button
            variant="ghost"
            onClick={() => router.push("/documents")}
            className="mb-6"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Documents
          </Button>

          {/* Document Header */}
          <div className="mb-6">
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl font-semibold text-gray-900">
                {document.filename}
              </h1>
              {getStatusBadge(document.status)}
            </div>
            <div className="flex items-center gap-4 text-gray-600">
              <div className="flex items-center gap-2">
                <Calendar className="h-4 w-4" />
                <span>{formatDate(document.uploaded_at)}</span>
              </div>
            </div>
          </div>

          {/* Error Message for Failed Documents */}
          {document.status === "failed" && document.error && (
            <Card className="mb-6 border-red-200 bg-red-50">
              <CardHeader>
                <CardTitle className="text-red-700">
                  Processing Failed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-sm text-red-600 whitespace-pre-wrap">
                  {document.error}
                </pre>
              </CardContent>
            </Card>
          )}

          {/* Processing Message */}
          {document.status === "processing" && (
            <Card className="mb-6 border-amber-200 bg-amber-50">
              <CardHeader>
                <CardTitle className="text-amber-700">
                  Processing in Progress
                </CardTitle>
                <CardDescription className="text-amber-600">
                  This document is currently being analyzed. Please check back
                  in a moment.
                </CardDescription>
              </CardHeader>
            </Card>
          )}

          {/* NLP Stats - Only show if completed and stats exist */}
          {document.status === "completed" && stats && (
            <div className="grid gap-6 md:grid-cols-2">
              {/* Vocabulary Statistics */}
              <Card className="p-6">
                <CardHeader className="pb-4">
                  <CardTitle className="text-lg font-semibold text-gray-900">
                    Vocabulary Statistics
                  </CardTitle>
                  <CardDescription className="text-gray-600">
                    Analysis of unique words and tokens
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6 pt-2">
                  <div className="space-y-1">
                    <div className="text-sm font-medium text-gray-600">
                      Vocabulary Size
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {stats.vocab_size.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-500">unique words</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm font-medium text-gray-600">
                      Total Tokens
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {stats.token_count.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-500">tokens</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm font-medium text-gray-600">
                      Type-Token Ratio
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {(stats.type_token_ratio * 100).toFixed(2)}%
                    </div>
                    <div className="text-sm text-gray-500">
                      Lexical diversity measure
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Sentiment Analysis */}
              <Card className="p-6">
                <CardHeader className="pb-4">
                  <CardTitle className="text-lg font-semibold text-gray-900">
                    Sentiment Analysis
                  </CardTitle>
                  <CardDescription className="text-gray-600">
                    Emotional tone detected in the text
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-2">
                  <div className="space-y-4">
                    {Object.entries(stats.doc_sentiment).map(
                      ([emotion, score]) => {
                        const scoreValue =
                          typeof score === "number" ? score : 0;
                        const percentage = (scoreValue * 100).toFixed(1);
                        const sentimentValues = Object.values(
                          stats.doc_sentiment,
                        ).map((v) => (typeof v === "number" ? v : 0));
                        const isHighest =
                          scoreValue === Math.max(...sentimentValues);

                        return (
                          <div key={emotion} className="space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="capitalize font-medium text-gray-700">
                                {emotion}
                              </span>
                              <span
                                className={`font-bold ${isHighest ? "text-violet-600" : "text-gray-600"}`}
                              >
                                {percentage}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-3">
                              <div
                                className={`h-3 rounded-full transition-all duration-300 ${
                                  isHighest ? "bg-violet-600" : "bg-gray-400"
                                }`}
                                style={{ width: `${percentage}%` }}
                              />
                            </div>
                          </div>
                        );
                      },
                    )}
                  </div>
                  <div className="mt-6 pt-4 border-t border-gray-200">
                    <div className="text-sm text-gray-500">
                      Method: {stats.sentiment_method}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </main>
      </div>
    </ProtectedRoute>
  );
}
