// API client for NLP Document Library backend
import { getAuthToken } from './auth';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Document {
	_id: string;
	filename: string;
	uploaded_at: string;
	status: 'processing' | 'completed' | 'failed';
	error?: string;
	stats?: DocumentStats;
}

export interface WordFrequency {
	lemma: string;
	count: number;
}

export interface NgramFrequency {
	ngram: string;
	count: number;
}

export interface SentenceSentiment {
	sentence: string;
	emotion: string;
	score: number;
}

export interface DocumentStats {
	vocab_size: number; // Unique words (vocabulary size)
	word_count: number; // Total words/tokens
	type_token_ratio: number;
	doc_sentiment: Record<string, number>;
	sentiment_method: string;
	word_frequencies?: WordFrequency[];
	ngrams?: {
		unigram?: NgramFrequency[];
		bigram?: NgramFrequency[];
		trigram?: NgramFrequency[];
	};
	pos_counts?: Record<string, number>;
	sentence_sentiment?: SentenceSentiment[];
	file?: string;
	sentence_count?: number;
	char_count?: number;
}

export interface UserDocuments {
	user_id: string;
	documents: Document[];
}

export interface UploadResponse {
	document_id: string;
	filename: string;
	processing_status: string;
	stats?: DocumentStats;
}

export interface UpdateDocumentRequest {
	filename?: string;
}

export interface DeleteDocumentResponse {
	success: boolean;
	message: string;
}

class ApiClient {
	private baseUrl: string;

	constructor(baseUrl: string = API_BASE_URL) {
		this.baseUrl = baseUrl;
	}

	private async request<T>(
		endpoint: string,
		options: RequestInit = {}
	): Promise<T> {
		const url = `${this.baseUrl}${endpoint}`;
		const token = getAuthToken();

		const headers: Record<string, string> = {
			'Content-Type': 'application/json',
			...(options.headers as Record<string, string>),
		};

		if (token) {
			headers.Authorization = `Bearer ${token}`;
		}

		const response = await fetch(url, {
			...options,
			headers,
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`API Error: ${response.status} - ${errorText}`);
		}

		return response.json();
	}

	// Get all documents for a user
	async getUserDocuments(userId: string): Promise<UserDocuments> {
		return this.request<UserDocuments>(`/documents/${userId}`);
	}

	// Get a specific document with stats
	async getDocument(userId: string, documentId: string): Promise<Document> {
		return this.request<Document>(`/documents/${userId}/${documentId}`);
	}

	// Upload a document
	async uploadDocument(
		userId: string,
		file: File,
		_onProgress?: (progress: number) => void
	): Promise<UploadResponse> {
		const formData = new FormData();
		// Note: user_id is extracted from JWT token on backend, don't send it in form data
		formData.append('file', file);

		const token = getAuthToken();
		const headers: Record<string, string> = {};
		if (token) {
			headers.Authorization = `Bearer ${token}`;
		}

		try {
			const response = await fetch(`${this.baseUrl}/documents/upload`, {
				method: 'POST',
				headers,
				body: formData,
			});

			if (!response.ok) {
				let errorMessage = `Upload failed (${response.status})`;
				try {
					const errorData = await response.json();
					// Extract detail from FastAPI error response
					if (errorData.detail) {
						errorMessage = errorData.detail;
					} else if (typeof errorData === 'string') {
						errorMessage = errorData;
					}
				} catch {
					// If JSON parsing fails, try text
					const errorText = await response.text();
					if (errorText) {
						try {
							const parsed = JSON.parse(errorText);
							errorMessage = parsed.detail || errorText;
						} catch {
							errorMessage = errorText || errorMessage;
						}
					}
				}
				throw new Error(errorMessage);
			}

			return response.json();
		} catch (error) {
			if (error instanceof TypeError && error.message.includes('fetch')) {
				throw new Error(
					'Failed to connect to server. Please check if the API is running.'
				);
			}
			throw error;
		}
	}

	async updateDocument(
		userId: string,
		documentId: string,
		filename: string
	): Promise<Document> {
		return this.request<Document>(
			`/documents/${userId}/${documentId}?filename=${encodeURIComponent(
				filename
			)}`,
			{
				method: 'PUT',
			}
		);
	}

	async deleteDocument(
		userId: string,
		documentId: string
	): Promise<DeleteDocumentResponse> {
		return this.request<DeleteDocumentResponse>(
			`/documents/${userId}/${documentId}`,
			{
				method: 'DELETE',
			}
		);
	}

	// Health check
	async healthCheck(): Promise<{ message: string; version: string }> {
		return this.request<{ message: string; version: string }>('/');
	}
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export individual functions for convenience
export const {
	getUserDocuments,
	getDocument,
	uploadDocument,
	updateDocument,
	deleteDocument,
	healthCheck,
} = apiClient;
