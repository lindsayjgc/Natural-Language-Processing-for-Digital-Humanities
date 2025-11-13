/**
 * Readability calculation utilities
 * Implements Flesch Reading Ease and Flesch-Kincaid Grade Level
 */

/**
 * Estimate the number of syllables in a word
 * Uses a simple heuristic: count vowel groups
 */
function estimateSyllables(word: string): number {
  word = word.toLowerCase().trim();
  if (word.length <= 3) return 1;
  
  // Remove silent 'e' at the end
  word = word.replace(/e$/, '');
  
  // Count vowel groups
  const vowelGroups = word.match(/[aeiouy]+/g);
  if (!vowelGroups) return 1;
  
  const syllableCount = vowelGroups.length;
  return Math.max(1, syllableCount);
}

/**
 * Calculate total syllables from word count
 * Uses average of 1.7 syllables per word as a reasonable estimate
 */
function estimateTotalSyllables(wordCount: number): number {
  return Math.round(wordCount * 1.7);
}

/**
 * Calculate Flesch Reading Ease score
 * @param totalWords Total number of words
 * @param totalSentences Total number of sentences
 * @param totalSyllables Total number of syllables (if not provided, will be estimated)
 * @returns Flesch Reading Ease score (0-100, higher is easier)
 */
export function calculateFleschReadingEase(
  totalWords: number,
  totalSentences: number,
  totalSyllables?: number
): number {
  if (totalWords === 0 || totalSentences === 0) {
    return 0;
  }

  const syllables = totalSyllables ?? estimateTotalSyllables(totalWords);
  
  // Average sentence length (ASL)
  const asl = totalWords / totalSentences;
  
  // Average syllables per word (ASW)
  const asw = syllables / totalWords;
  
  // Flesch Reading Ease formula
  const score = 206.835 - (1.015 * asl) - (84.6 * asw);
  
  // Clamp to 0-100 range
  return Math.max(0, Math.min(100, score));
}

/**
 * Calculate Flesch-Kincaid Grade Level
 * @param totalWords Total number of words
 * @param totalSentences Total number of sentences
 * @param totalSyllables Total number of syllables (if not provided, will be estimated)
 * @returns Flesch-Kincaid Grade Level (typically 0-20)
 */
export function calculateFleschKincaidGradeLevel(
  totalWords: number,
  totalSentences: number,
  totalSyllables?: number
): number {
  if (totalWords === 0 || totalSentences === 0) {
    return 0;
  }

  const syllables = totalSyllables ?? estimateTotalSyllables(totalWords);
  
  // Average sentence length (ASL)
  const asl = totalWords / totalSentences;
  
  // Average syllables per word (ASW)
  const asw = syllables / totalWords;
  
  // Flesch-Kincaid Grade Level formula
  const gradeLevel = (0.39 * asl) + (11.8 * asw) - 15.59;
  
  // Clamp to reasonable range (0-20)
  return Math.max(0, Math.min(20, gradeLevel));
}

/**
 * Format readability score for display
 */
export function formatReadabilityScore(score: number, decimals: number = 1): string {
  if (isNaN(score) || !isFinite(score)) {
    return "N/A";
  }
  return score.toFixed(decimals);
}

