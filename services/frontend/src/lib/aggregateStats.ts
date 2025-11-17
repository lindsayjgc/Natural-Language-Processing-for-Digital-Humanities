/**
 * Statistics aggregation utilities
 * Aggregates document statistics across multiple documents
 */

import type { DocumentStats, WordFrequency, NgramFrequency, SentenceSentiment } from "./api";

export interface AggregatedStats {
  // Summary stats
  totalTokens: number;
  totalCharacters: number;
  totalSentences: number;
  totalDocuments: number;
  
  // Vocabulary
  vocabSize: number;
  typeTokenRatio: number;
  
  // Sentiment
  docSentiment: Record<string, number>;
  
  // Word frequencies (merged and sorted)
  wordFrequencies: WordFrequency[];
  
  // N-grams (merged and sorted)
  ngrams: {
    unigram?: NgramFrequency[];
    bigram?: NgramFrequency[];
    trigram?: NgramFrequency[];
  };
  
  // POS counts (merged)
  posCounts: Record<string, number>;
  
  // Sentence sentiment (merged, sorted by score)
  sentenceSentiment: SentenceSentiment[];
}

/**
 * Aggregate statistics across multiple documents
 */
export function aggregateStats(documents: Array<{ stats: DocumentStats }>): AggregatedStats {
  if (documents.length === 0) {
    return {
      totalTokens: 0,
      totalCharacters: 0,
      totalSentences: 0,
      totalDocuments: 0,
      vocabSize: 0,
      typeTokenRatio: 0,
      docSentiment: {},
      wordFrequencies: [],
      ngrams: {},
      posCounts: {},
      sentenceSentiment: [],
    };
  }

  // Summable metrics
  let totalTokens = 0;
  let totalSentences = 0;
  let totalCharacters = 0;
  
  // For weighted averages
  let totalWeightedTokens = 0;
  
  // For sentiment aggregation (weighted by token count)
  const sentimentSums: Record<string, number> = {};
  const sentimentWeights: Record<string, number> = {};
  
  // For word frequency aggregation
  const wordFrequencyMap = new Map<string, number>();
  
  // For n-gram aggregation
  const ngramMaps: {
    unigram?: Map<string, number>;
    bigram?: Map<string, number>;
    trigram?: Map<string, number>;
  } = {
    unigram: new Map(),
    bigram: new Map(),
    trigram: new Map(),
  };
  
  // For POS count aggregation
  const posCountMap = new Map<string, number>();
  
  // For sentence sentiment aggregation
  const allSentenceSentiments: SentenceSentiment[] = [];

  // Process each document
  for (const { stats } of documents) {
    if (!stats) continue;
    
    // Sum words
    const docWords = stats.word_count || 0;
    totalTokens += docWords;
    totalWeightedTokens += docWords;
    
    // Sum sentences (from sentence_sentiment if available, or use sentence_count)
    if (stats.sentence_sentiment && stats.sentence_sentiment.length > 0) {
      totalSentences += stats.sentence_sentiment.length;
      // Collect sentence sentiments
      allSentenceSentiments.push(...stats.sentence_sentiment);
    } else if (stats.sentence_count) {
      totalSentences += stats.sentence_count;
    } else {
      // Estimate: average 15 words per sentence
      totalSentences += Math.max(1, Math.round(docWords / 15));
    }
    
    // Estimate characters from word count
    // Average word length is ~5 characters, plus spaces between words
    // Estimate: (average_word_length * word_count) + (word_count - 1) for spaces
    // Simplified: ~5.5 characters per word on average (including spaces)
    totalCharacters += stats.char_count || Math.round(docWords * 5.5);
    
    // Aggregate sentiment (weighted by word count)
    if (stats.doc_sentiment && docWords > 0) {
      for (const [emotion, score] of Object.entries(stats.doc_sentiment)) {
        if (typeof score === "number" && isFinite(score)) {
          sentimentSums[emotion] = (sentimentSums[emotion] || 0) + (score * docWords);
          sentimentWeights[emotion] = (sentimentWeights[emotion] || 0) + docWords;
        }
      }
    }
    
    // Aggregate word frequencies
    if (stats.word_frequencies) {
      for (const wf of stats.word_frequencies) {
        const currentCount = wordFrequencyMap.get(wf.lemma) || 0;
        wordFrequencyMap.set(wf.lemma, currentCount + wf.count);
      }
    }
    
    // Aggregate n-grams
    if (stats.ngrams) {
      if (stats.ngrams.unigram) {
        for (const ngram of stats.ngrams.unigram) {
          const currentCount = ngramMaps.unigram!.get(ngram.ngram) || 0;
          ngramMaps.unigram!.set(ngram.ngram, currentCount + ngram.count);
        }
      }
      if (stats.ngrams.bigram) {
        for (const ngram of stats.ngrams.bigram) {
          const currentCount = ngramMaps.bigram!.get(ngram.ngram) || 0;
          ngramMaps.bigram!.set(ngram.ngram, currentCount + ngram.count);
        }
      }
      if (stats.ngrams.trigram) {
        for (const ngram of stats.ngrams.trigram) {
          const currentCount = ngramMaps.trigram!.get(ngram.ngram) || 0;
          ngramMaps.trigram!.set(ngram.ngram, currentCount + ngram.count);
        }
      }
    }
    
    // Aggregate POS counts
    if (stats.pos_counts) {
      for (const [pos, count] of Object.entries(stats.pos_counts)) {
        const currentCount = posCountMap.get(pos) || 0;
        posCountMap.set(pos, currentCount + count);
      }
    }
  }

  // Calculate weighted average sentiment
  const aggregatedSentiment: Record<string, number> = {};
  for (const emotion of Object.keys(sentimentSums)) {
    const weight = sentimentWeights[emotion];
    if (weight > 0) {
      aggregatedSentiment[emotion] = sentimentSums[emotion] / weight;
    }
  }

  // Convert word frequency map to sorted array
  const wordFrequencies: WordFrequency[] = Array.from(wordFrequencyMap.entries())
    .map(([lemma, count]) => ({ lemma, count }))
    .sort((a, b) => b.count - a.count);

  // Calculate vocab size (unique words)
  const vocabSize = wordFrequencyMap.size;

  // Calculate type-token ratio
  const typeTokenRatio = totalTokens > 0 ? vocabSize / totalTokens : 0;

  // Convert n-gram maps to sorted arrays
  const aggregatedNgrams: {
    unigram?: NgramFrequency[];
    bigram?: NgramFrequency[];
    trigram?: NgramFrequency[];
  } = {};
  
  if (ngramMaps.unigram && ngramMaps.unigram.size > 0) {
    aggregatedNgrams.unigram = Array.from(ngramMaps.unigram.entries())
      .map(([ngram, count]) => ({ ngram, count }))
      .sort((a, b) => b.count - a.count);
  }
  
  if (ngramMaps.bigram && ngramMaps.bigram.size > 0) {
    aggregatedNgrams.bigram = Array.from(ngramMaps.bigram.entries())
      .map(([ngram, count]) => ({ ngram, count }))
      .sort((a, b) => b.count - a.count);
  }
  
  if (ngramMaps.trigram && ngramMaps.trigram.size > 0) {
    aggregatedNgrams.trigram = Array.from(ngramMaps.trigram.entries())
      .map(([ngram, count]) => ({ ngram, count }))
      .sort((a, b) => b.count - a.count);
  }

  // Convert POS count map to object
  const posCounts: Record<string, number> = Object.fromEntries(posCountMap.entries());

  // Sort sentence sentiments by score (descending) - show all sentences
  const sortedSentenceSentiments = allSentenceSentiments
    .sort((a, b) => b.score - a.score);

  return {
    totalTokens,
    totalCharacters,
    totalSentences,
    totalDocuments: documents.length,
    vocabSize,
    typeTokenRatio,
    docSentiment: aggregatedSentiment,
    wordFrequencies,
    ngrams: aggregatedNgrams,
    posCounts,
    sentenceSentiment: sortedSentenceSentiments,
  };
}

/**
 * Transform aggregated sentiment into polarity data (positive/neutral/negative)
 */
export function sentimentToPolarity(
  docSentiment: Record<string, number>
): Array<{ name: string; value: number; color: string }> {
  // Map emotions to polarity categories
  const positiveEmotions = ["joy", "happy", "positive", "surprise"];
  const negativeEmotions = ["sadness", "sad", "negative", "anger", "angry", "fear", "disgust"];
  const neutralEmotions = ["neutral"];

  let positive = 0;
  let negative = 0;
  let neutral = 0;
  let total = 0;

  for (const [emotion, score] of Object.entries(docSentiment)) {
    const emotionLower = emotion.toLowerCase();
    const normalizedScore = Math.abs(score);
    
    if (positiveEmotions.some(e => emotionLower.includes(e))) {
      positive += normalizedScore;
    } else if (negativeEmotions.some(e => emotionLower.includes(e))) {
      negative += normalizedScore;
    } else if (neutralEmotions.some(e => emotionLower.includes(e))) {
      neutral += normalizedScore;
    } else {
      // Default to neutral for unknown emotions
      neutral += normalizedScore;
    }
    total += normalizedScore;
  }

  // Normalize to percentages
  if (total === 0) {
    return [
      { name: "Positive", value: 0, color: "#10B981" },
      { name: "Neutral", value: 100, color: "#D1D5DB" },
      { name: "Negative", value: 0, color: "#EF4444" },
    ];
  }

  const positivePercent = (positive / total) * 100;
  const negativePercent = (negative / total) * 100;
  const neutralPercent = (neutral / total) * 100;

  return [
    { name: "Positive", value: Math.round(positivePercent), color: "#10B981" },
    { name: "Neutral", value: Math.round(neutralPercent), color: "#D1D5DB" },
    { name: "Negative", value: Math.round(negativePercent), color: "#EF4444" },
  ];
}

/**
 * Transform aggregated sentiment into analysis data (specific emotions)
 */
export function sentimentToAnalysis(
  docSentiment: Record<string, number>
): Array<{ name: string; value: number; color: string }> {
  // Map of emotion names to colors
  const emotionColors: Record<string, string> = {
    joy: "#10B981",
    happy: "#10B981",
    positive: "#10B981",
    surprise: "#F59E0B",
    sadness: "#3B82F6",
    sad: "#3B82F6",
    anger: "#EF4444",
    angry: "#EF4444",
    fear: "#8B5CF6",
    disgust: "#F97316",
    neutral: "#9CA3AF",
  };

  // Convert to array and normalize to percentages
  const entries = Object.entries(docSentiment)
    .map(([emotion, score]) => ({
      name: emotion
        .toLowerCase()
        .replace(/_/g, " ")
        .replace(/\b\w/g, (l) => l.toUpperCase()),
      value: Math.abs(score),
      emotion: emotion.toLowerCase(),
    }))
    .filter((item) => item.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 6); // Top 6 emotions

  // Calculate total for normalization
  const total = entries.reduce((sum, item) => sum + item.value, 0);

  if (total === 0) {
    return [];
  }

  // Normalize to percentages and assign colors
  return entries.map((item) => {
    const percent = (item.value / total) * 100;
    // Find matching color
    const colorKey = Object.keys(emotionColors).find((key) =>
      item.emotion.includes(key)
    );
    const color = colorKey ? emotionColors[colorKey] : "#64748B";

    return {
      name: item.name,
      value: Math.round(percent),
      color,
    };
  });
}

