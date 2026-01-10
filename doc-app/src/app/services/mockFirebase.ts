export class MockAuth {
  currentUser: any = null;
  listeners: Function[] = [];

  onAuthStateChanged(cb: Function) {
    this.listeners.push(cb);
    cb(this.currentUser);
    return () => {};
  }

  async signInWithPopup() {
    this.currentUser = {
      uid: "demo",
      email: "demo@example.com",
      displayName: "Demo",
    };
    this.listeners.forEach((l) => l(this.currentUser));
  }
}

export const mockAuth = new MockAuth();
