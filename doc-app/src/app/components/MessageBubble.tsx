"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";

export default function MessageBubble({ message }: any) {
  const isUser = message.role === "user";
  const isThinking = message.isThinking;

  return (
    <div className={`my-3 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-3xl rounded-xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-gray-900 text-gray-200"
        }`}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={{
            code({ className, children }) {
              const isBlock = Boolean(className);

              return isBlock ? (
                <pre className="bg-black rounded-lg p-4 my-3 overflow-x-auto">
                  <code className={className}>{children}</code>
                </pre>
              ) : (
                <code className="bg-black/40 px-1 py-0.5 rounded">
                  {children}
                </code>
              );
            },
            a({ href, children }) {
              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-indigo-400 underline"
                >
                  {children}
                </a>
              );
            },
          }}
        >
          {message.content}
        </ReactMarkdown>


        {message.fileUrl && (
          <div className="mt-2 text-xs">
            📎{" "}
            <a
              href={message.fileUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-400 underline"
            >
              View attachment
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
