"use client";

import {
  addDoc,
  collection,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  where,
  deleteDoc,
  updateDoc,
  doc,
} from "firebase/firestore";
import { useEffect, useState } from "react";
import { db } from "../services/firebase";


export function useChats(userId?: string) {
  const [chats, setChats] = useState<any[]>([]);

  useEffect(() => {
    if (!userId) return;

    const q = query(
      collection(db, "chats"),
      where("userId", "==", userId),
      orderBy("updatedAt", "desc")
    );

    return onSnapshot(q, (snap) => {
      setChats(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
    });
  }, [userId]);

  const createChat = async () => {
    const ref = await addDoc(collection(db, "chats"), {
      userId,
      title: "New Chat",
      updatedAt: serverTimestamp(),
    });
    return ref.id;
  };

  const updateChat = async (id: string, data: any) =>
    updateDoc(doc(db, "chats", id), {
      ...data,
      updatedAt: serverTimestamp(),
    });

  const deleteChat = async (chatId: string) => {
    await deleteDoc(doc(db, "chats", chatId));
  };

  const togglePin = async (chatId: string, pinned: boolean) => {
    await updateDoc(doc(db, "chats", chatId), {
      pinned,
    });
  };

  return { chats, loading: false,createChat, updateChat,deleteChat,togglePin };
}
