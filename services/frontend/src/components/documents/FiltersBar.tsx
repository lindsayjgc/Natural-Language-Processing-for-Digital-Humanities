"use client";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

export function FiltersBar() {
    return (
        <div className="mt-6 flex flex-wrap items-center gap-3">
            <div className="relative grow sm:grow-0 sm:w-80" />
            <div className="ml-auto flex items-center gap-3">
                <Select>
                    <SelectTrigger className="w-[140px] bg-white border-black/10 text-gray-900">
                        <SelectValue placeholder="Sort by" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="uploaded">Uploaded</SelectItem>
                        <SelectItem value="name">Name</SelectItem>
                        <SelectItem value="status">Status</SelectItem>
                    </SelectContent>
                </Select>
                <Separator orientation="vertical" className="h-6 bg-black/10" />
                <Button className="bg-white border border-black/10 text-gray-900 hover:bg-gray-50">Filters</Button>
            </div>
        </div>
    );
}
