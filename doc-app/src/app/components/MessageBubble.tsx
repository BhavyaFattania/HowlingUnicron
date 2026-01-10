"use client";

export default function MessageBubble({ message }: any) {
  const isUser = message.role === "user";

  return (
    <div className={`my-2 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`px-4 py-2 rounded max-w-md ${
          isUser ? "bg-indigo-600 text-white" : "bg-gray-200"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
