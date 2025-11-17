"use client";

import { useCallback, useEffect, useState } from "react";
import { DocumentsTable } from "@/components/documents/DocumentsTable";
import { FiltersBar } from "@/components/documents/FiltersBar";
import { UploadCard } from "@/components/documents/UploadCard";
import { useAuth } from "@/contexts/AuthContext";
import { apiClient, type Document } from "@/lib/api";

export function DocumentsView() {
  const [_files, _setFiles] = useState<File[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);

  // Mock user ID for now - in a real app this would come from auth
  // const userId = "demo_user";
  const { user } = useAuth();
  const userId = user?.id;

  const fetchDocuments = useCallback(async () => {
    if (!userId) return; // prevent calling with undefined
    try {
      setLoading(true);
      setError(null);
      const response = await apiClient.getUserDocuments(userId);
      setDocuments(response.documents);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to fetch documents",
      );
    } finally {
      setLoading(false);
    }
  }, [userId]);

  // Fetch documents on component mount
  useEffect(() => {
    if (!userId) return;
    fetchDocuments();
  }, [fetchDocuments, userId]);

  const handleFileUpload = async (files: File[]) => {
    if (files.length === 0 || !userId) return;

    setUploading(true);
    setError(null);
    setProcessingStatus(null);

    try {
      // Upload each file and wait for processing to complete
      for (const file of files) {
        // Check file type and provide helpful error before upload
        const fileName = file.name.toLowerCase();
        if (fileName.endsWith('.pdf')) {
          throw new Error(
            'PDF files are not directly supported. Please convert your PDF to a .txt file before uploading. ' +
            'You can use online converters or copy the text content into a .txt file.'
          );
        }

        const uploadResponse = await apiClient.uploadDocument(userId, file);

        // Poll for document completion if it's still processing
        if (uploadResponse.processing_status === "processing") {
          await pollForDocumentCompletion(uploadResponse.document_id);
        }
      }

      // Refresh the document list
      await fetchDocuments();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to upload documents";
      
      // Provide user-friendly messages for common errors
      if (errorMessage.includes('PDF') || errorMessage.includes('textract')) {
        setError(
          'PDF files are not directly supported. Please convert your PDF to a .txt file before uploading. ' +
          'You can use online converters or copy the text content into a .txt file.'
        );
      } else {
        setError(errorMessage);
      }
    } finally {
      setUploading(false);
      setProcessingStatus(null);
    }
  };

  const pollForDocumentCompletion = async (
    documentId: string,
    maxAttempts = 30,
  ) => {
    if (!userId) throw new Error("User ID is required to poll documents");
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        setProcessingStatus(
          `Processing document... (${attempt + 1}/${maxAttempts})`,
        );
        const document = await apiClient.getDocument(userId, documentId);

        if (document.status === "completed") {
          setProcessingStatus("Document processed successfully!");
          return document;
        } else if (document.status === "failed") {
          throw new Error(document.error || "Document processing failed");
        }

        // Wait 1 second before next poll
        await new Promise((resolve) => setTimeout(resolve, 1000));
      } catch (err) {
        // If it's a 404, the document might not be ready yet, continue polling
        if (err instanceof Error && err.message.includes("404")) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          continue;
        }
        throw err;
      }
    }

    throw new Error("Document processing timed out after 30 seconds");
  };

  const handleUpdateDocument = async (id: string, newName: string) => {
    if (!userId) return;
    
    try {
      setError(null);
      await apiClient.updateDocument(userId, id, newName);
      await fetchDocuments();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to update document",
      );
    }
  };

  const handleDeleteDocument = async (id: string) => {
    if (!userId) return;
    
    try {
      setError(null);
      await apiClient.deleteDocument(userId, id);
      await fetchDocuments();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to delete document",
      );
    }
  };

  return (
    <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
      <main className="mx-auto max-w-6xl px-4 py-10">
        <h1 className="text-3xl font-semibold mb-2">Documents</h1>
        <p className="text-gray-600 mb-8">
          Upload, manage, and analyze your literary texts.
        </p>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <UploadCard onDropFiles={handleFileUpload} uploading={uploading} />

        {uploading && (
          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg text-blue-700">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
              {processingStatus ||
                "Processing your document... This may take a few moments."}
            </div>
          </div>
        )}
        
        <FiltersBar />
        <DocumentsTable
          documents={documents}
          loading={loading}
          onRefresh={fetchDocuments}
          onUpdateDocument={handleUpdateDocument}
          onDeleteDocument={handleDeleteDocument}
        />
      </main>
    </div>
  );
}
