"use client";

import { useState, useEffect, useRef } from "react";
import { Paperclip, Mic, ArrowUp, X } from "lucide-react";

const SUGGESTIONS = [
  "Explain quantum computing",
  "Write a poem about rain",
  "Debug my React hook",
  "Plan a trip to Tokyo",
];

export default function ChatInput({ onSend }: any) {
  const [text, setText] = useState("");
  const [suggestionIndex, setSuggestionIndex] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setSuggestionIndex((prev) => (prev + 1) % SUGGESTIONS.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSend = () => {
    if (!text.trim() && !selectedFile) return;
    onSend(text, selectedFile);
    setText("");
    setSelectedFile(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-6">
      <div className={`text-center text-xs font-medium text-gray-500 mb-3 transition-opacity duration-700 ${isFocused ? "opacity-0" : "opacity-100"}`}>
        <span className="bg-gray-800 px-3 py-1 rounded-full border border-gray-700 text-gray-300">
          ✨ Try asking: "{SUGGESTIONS[suggestionIndex]}"
        </span>
      </div>

      <div className={`relative bg-gray-800 transition-all duration-200 rounded-2xl flex flex-col ${isFocused ? "border border-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.3)]" : "border border-gray-700"}`}>
        
        {selectedFile && (
          <div className="mx-4 mt-3 flex items-center gap-2 bg-gray-700 text-gray-200 px-3 py-1 rounded-lg w-fit text-sm border border-gray-600">
            <Paperclip className="w-4 h-4" />
            <span className="truncate max-w-[150px]">{selectedFile.name}</span>
            <button onClick={() => setSelectedFile(null)} className="hover:text-white"><X className="w-4 h-4" /></button>
          </div>
        )}

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything..."
          className="w-full bg-transparent border-none px-4 py-4 text-gray-100 focus:ring-0 resize-none max-h-48 overflow-y-auto placeholder-gray-500"
          rows={1}
          style={{ minHeight: "60px" }}
        />

        <div className="flex justify-between items-center px-2 pb-2">
          <div className="flex gap-1">
            <input type="file" ref={fileInputRef} onChange={(e) => e.target.files?.[0] && setSelectedFile(e.target.files[0])} className="hidden" />
            <button onClick={() => fileInputRef.current?.click()} className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-full transition-colors">
              <Paperclip className="w-5 h-5" />
            </button>
            <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-full transition-colors">
              <Mic className="w-5 h-5" />
            </button>
          </div>

          <button
            onClick={handleSend}
            disabled={!text.trim() && !selectedFile}
            className={`p-2 rounded-lg transition-all duration-200 ${
              text.trim() || selectedFile ? "bg-indigo-600 text-white shadow-lg hover:bg-indigo-500" : "bg-gray-700 text-gray-500 cursor-not-allowed"
            }`}
          >
            <ArrowUp className="w-5 h-5" />
          </button>
        </div>
      </div>
      <div className="text-center mt-2">
        <p className="text-[10px] text-gray-600">AI can make mistakes. Check important info.</p>
      </div>
    </div>
  );
}