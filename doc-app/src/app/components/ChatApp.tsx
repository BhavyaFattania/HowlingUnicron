"use client";

import { useState } from "react";
import { useAuth } from "../contexts/AuthContext";
import { useChats } from "../hooks/useChats";
import LeftSidebar from "./LeftSidebar";
import RightSidebar from "./RightSidebar";
import ChatWindow from "./ChatWindow";
import LoginPage from "./LoginPage";
import { Loader2 } from "lucide-react";

export default function ChatApp() {
  const { user, loading: authLoading } = useAuth();
  const { chats, loading, createChat, updateChat,deleteChat, togglePin} = useChats(user?.uid);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);

  const [temperature, setTemperature] = useState(0.7);
  const [webSearch, setWebSearch] = useState(false);

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  if (!user) return <LoginPage />;

  return (
    <div className="flex h-screen bg-black">
      <LeftSidebar
        chats={chats}
        loading={loading}
        currentChatId={currentChatId}
        onSelectChat={setCurrentChatId}
        onNewChat={async () => {
          const id = await createChat();
          setCurrentChatId(id);
        }}
        onDeleteChat={deleteChat}
        onTogglePin={togglePin}
      />

      <ChatWindow
        chatId={currentChatId}
        onStartNewChat={async () => {
          const id = await createChat();
          setCurrentChatId(id);
        }}
        onUpdateChat={updateChat}
        temperature={temperature}
        webSearch={webSearch}
      />

      <RightSidebar
        temperature={temperature}
        setTemperature={setTemperature}
        webSearch={webSearch}
        setWebSearch={setWebSearch}
      />
    </div>
  );
}
