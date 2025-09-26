"use client";

import { useState } from "react";
import { UploadCard } from "@/components/documents/UploadCard";
import { FiltersBar } from "@/components/documents/FiltersBar";
import { DocumentsTable } from "@/components/documents/DocumentsTable";

export function DocumentsView() {
    const [files, setFiles] = useState<File[]>([]);

    return (
        <div className="min-h-[calc(100vh-56px)] bg-gray-50 text-gray-900">
            <main className="mx-auto max-w-6xl px-4 py-10">
                <h1 className="text-3xl font-semibold mb-2">Documents</h1>
                <p className="text-gray-600 mb-8">Upload, manage, and analyze your literary texts.</p>
                <UploadCard onDropFiles={(dropped) => setFiles(dropped)} />
                <FiltersBar />
                <DocumentsTable />
            </main>
        </div>
    );
}
