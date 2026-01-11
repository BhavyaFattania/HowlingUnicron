"use client";

import { useState } from "react";
import { useAuth } from "../contexts/AuthContext";

import {
  MessageSquare,
  Plus,
  LogOut,
  MoreVertical,
  Pin,
  Trash2,
  User,
} from "lucide-react";

export default function LeftSidebar({
  chats,
  loading,
  currentChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onTogglePin
}: any) {
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const pinnedChats = chats.filter((c: any) => c.pinned);
  const recentChats = chats.filter((c: any) => !c.pinned);
  const { logout } = useAuth();


  return (
    <div className="w-64 bg-black border-r border-gray-800 flex flex-col h-screen text-gray-300">
      
      <div className="p-4">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-3 px-4 py-3 bg-gray-900 hover:bg-gray-800 border border-gray-800 rounded-lg transition-all text-sm font-medium text-white"
        >
          <Plus className="w-5 h-5 text-indigo-500" />
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 space-y-1">
        {pinnedChats.length > 0 && (
          <>
            <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase">
              Pinned
            </div>
            {pinnedChats.map((chat: any) => (
              <ChatRow
                key={chat.id}
                chat={chat}
                active={currentChatId === chat.id}
                onSelectChat={onSelectChat}
                openMenuId={openMenuId}
                setOpenMenuId={setOpenMenuId}
                onPin={() => onTogglePin(chat.id, !chat.pinned)}
                onDeleteChat={onDeleteChat}
              />
            ))}
          </>
        )}

        <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase">
          Recent
        </div>
        {recentChats.map((chat: any) => (
          <ChatRow
            key={chat.id}
            chat={chat}
            active={currentChatId === chat.id}
            onSelectChat={onSelectChat}
            openMenuId={openMenuId}
            setOpenMenuId={setOpenMenuId}
            onPin={() => onTogglePin(chat.id, !chat.pinned)}
            onDeleteChat={onDeleteChat}
          />
        ))}
      </div>

      <div className="p-4 border-t border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white">
            <User className="w-4 h-4" />
          </div>
          <span className="text-sm text-gray-400">Account</span>
        </div>

        <button
          onClick={logout}
          className="text-gray-500 hover:text-red-400"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}



function ChatRow({
  chat,
  active,
  onSelectChat,
  openMenuId,
  setOpenMenuId,
  onPin,
  onDeleteChat
}: any) {
  const isMenuOpen = openMenuId === chat.id;

  return (
    <div
      className={`group relative flex items-center gap-3 px-4 py-3 text-sm rounded-lg ${
        active
          ? "bg-indigo-600/10 text-indigo-400"
          : "hover:bg-gray-900 text-gray-400 hover:text-gray-200"
      }`}
    >
      <button
        onClick={() => onSelectChat(chat.id)}
        className="flex items-center gap-3 flex-1 text-left"
      >
        <MessageSquare className="w-4 h-4 shrink-0" />
        <span className="truncate">{chat.title || "New Conversation"}</span>
      </button>

      <button
        onClick={(e) => {
          e.stopPropagation();
          setOpenMenuId(isMenuOpen ? null : chat.id);
        }}
        className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-gray-300"
      >
        <MoreVertical className="w-4 h-4" />
      </button>

      {isMenuOpen && (
        <div className="absolute right-2 top-10 z-20 w-36 bg-gray-900 border border-gray-800 rounded-lg shadow-lg">
          <button
            onClick={() => {
              onPin();
              setOpenMenuId(null);
            }}
            className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-800"
          >
            <Pin className="w-4 h-4" />
            {chat.pinned ? "Unpin" : "Pin"}
          </button>
          <button
            onClick={() => {
              onDeleteChat(chat.id);
              setOpenMenuId(null);
            }}
            className="w-full flex items-center gap-2 px-3 py-2 text-red-400 hover:bg-red-500/10"
          >
            <Trash2 className="w-4 h-4" />
            Delete
          </button>
        </div>
      )}
    </div>
  );
}
