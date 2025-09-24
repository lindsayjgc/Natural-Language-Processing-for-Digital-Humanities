"use client";

import Link from "next/link";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Separator } from "@/components/ui/separator";

export function TopBar() {
    return (
        <div className="w-full sticky top-0 z-50 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80 border-b border-black/10">
            <div className="mx-auto max-w-6xl px-4 h-14 flex items-center gap-3">
                <Link href="/documents" className="flex items-center gap-2">
                    <span className="text-gray-900 font-semibold">LitLens</span>
                </Link>
                <Separator decorative orientation="vertical" className="h-6 bg-black/10" />
                <nav className="hidden md:flex items-center gap-1 text-sm">
                    <Link href="/documents" className="text-gray-900 px-3 py-1 rounded-md font-medium">
                        Documents
                    </Link>
                </nav>
                <div className="ml-auto flex items-center gap-3">
                    <Button className="hidden sm:inline-flex bg-violet-600 hover:bg-violet-700 text-white">New Text</Button>
                    <DropdownMenu>
                        <DropdownMenuTrigger className="outline-none">
                            <Avatar className="size-8 ring-1 ring-black/10">
                                <AvatarImage src="" alt="User" />
                                <AvatarFallback className="bg-gray-100 text-gray-700">U</AvatarFallback>
                            </Avatar>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="min-w-48">
                            <DropdownMenuLabel>Account</DropdownMenuLabel>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem>Profile</DropdownMenuItem>
                            <DropdownMenuItem>Settings</DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem>Sign out</DropdownMenuItem>
                        </DropdownMenuContent>
                    </DropdownMenu>
                </div>
            </div>
        </div>
    );
}
