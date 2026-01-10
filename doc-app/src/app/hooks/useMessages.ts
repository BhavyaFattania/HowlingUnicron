"use client";

import {
  addDoc,
  collection,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  where,
} from "firebase/firestore";
import { useEffect, useState } from "react";
import { db } from "../services/firebase";

export function useMessages(chatId?: string) {
  const [messages, setMessages] = useState<any[]>([]);

  useEffect(() => {
    if (!chatId) return;

    const q = query(
      collection(db, "messages"),
      where("chatId", "==", chatId),
      orderBy("createdAt")
    );

    return onSnapshot(q, (snap) => {
      setMessages(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
    });
  }, [chatId]);

  const addMessage = async (data: any) =>
    addDoc(collection(db, "messages"), {
      ...data,
      chatId,
      createdAt: serverTimestamp(),
    });

  return { messages, addMessage, loading: false };
}
