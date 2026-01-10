import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { storage } from "./firebase";

export async function uploadFile(path: string, file: File) {
  const r = ref(storage, path);
  await uploadBytes(r, file);
  return getDownloadURL(r);
}
