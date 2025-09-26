"use client";

import { useCallback, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { UploadCloud } from "lucide-react";

type UploadCardProps = {
    onBrowse?: () => void;
    onDropFiles?: (files: File[]) => void;
};

export function UploadCard({ onBrowse, onDropFiles }: UploadCardProps) {
    const handleDrop = useCallback(
        (event: React.DragEvent<HTMLDivElement>) => {
            event.preventDefault();
            const dropped = Array.from(event.dataTransfer.files ?? []);
            if (dropped.length && onDropFiles) onDropFiles(dropped);
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
        if (files.length && onDropFiles) onDropFiles(files);
        // reset so the same file can be selected again
        e.currentTarget.value = "";
    };

    return (
        <Card className="bg-white border-black/10 shadow-sm">
            <CardHeader>
                <CardTitle className="text-gray-900">My Documents</CardTitle>
                <CardDescription className="text-gray-600">
                    Upload, manage, and analyze your literary texts.
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleDrop}
                    className="rounded-xl border border-dashed border-black/15 bg-gray-50 p-10 text-center"
                >
                    <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-full bg-violet-50 text-violet-600">
                        <UploadCloud aria-hidden className="size-6" />
                    </div>
                    <div className="text-gray-900 font-medium">Drag and drop files to upload</div>
                    <div className="text-gray-600 text-sm mb-4">or click to browse your computer</div>
                    <Button onClick={handleBrowseClick} className="bg-violet-600 hover:bg-violet-700 text-white">
                        Browse Files
                    </Button>
                    <input
                        ref={inputRef}
                        type="file"
                        multiple
                        className="sr-only"
                        onChange={handleInputChange}
                    />
                </div>
            </CardContent>
        </Card>
    );
}
