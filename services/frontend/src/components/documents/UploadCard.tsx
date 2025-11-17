"use client";

import { UploadCloud } from "lucide-react";
import { useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

type UploadCardProps = {
  onBrowse?: () => void;
  onDropFiles?: (files: File[]) => void;
  uploading?: boolean;
};

export function UploadCard({
  onBrowse,
  onDropFiles,
  uploading = false,
}: UploadCardProps) {
  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const dropped = Array.from(event.dataTransfer.files ?? []);
      // Filter out unsupported file types (only allow .txt files)
      const txtFiles = dropped.filter(file => {
        const fileName = file.name.toLowerCase();
        return fileName.endsWith('.txt') || fileName.endsWith('.text');
      });
      
      if (txtFiles.length && onDropFiles) {
        onDropFiles(txtFiles);
      } else if (dropped.length > txtFiles.length) {
        // Some files were filtered out - could show a warning here if needed
        if (txtFiles.length > 0 && onDropFiles) {
          onDropFiles(txtFiles);
        }
      }
    },
    [onDropFiles],
  );
  const inputRef = useRef<HTMLInputElement | null>(null);
  const handleBrowseClick = () => {
    if (onBrowse) onBrowse();
    inputRef.current?.click();
  };
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    // Filter to only .txt files (though accept attribute should handle this)
    const txtFiles = files.filter(file => {
      const fileName = file.name.toLowerCase();
      return fileName.endsWith('.txt') || fileName.endsWith('.text');
    });
    if (txtFiles.length && onDropFiles) onDropFiles(txtFiles);
    // reset so the same file can be selected again
    e.currentTarget.value = "";
  };

  return (
    <Card>
      <CardContent className="p-5">
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-violet-400 transition-colors cursor-pointer w-full"
        >
          <div className="mx-auto mb-6 flex size-16 items-center justify-center rounded-full bg-violet-100 text-violet-600">
            <UploadCloud aria-hidden className="size-8" />
          </div>
          <div className="text-lg font-medium text-gray-900 mb-2">
            Drag and drop files to upload
          </div>
          <div className="text-gray-600 mb-8">
            or click to browse your computer
            <br />
            <span className="text-sm text-gray-500 mt-2 block">
              Supported formats: .txt files only
            </span>
          </div>
          <Button
            onClick={handleBrowseClick}
            disabled={uploading}
            size="lg"
            className="bg-violet-600 hover:bg-violet-700"
          >
            {uploading ? "Uploading..." : "Browse Files"}
          </Button>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".txt,.text"
            className="sr-only"
            onChange={handleInputChange}
          />
        </div>
      </CardContent>
    </Card>
  );
}
