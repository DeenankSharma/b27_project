import { BrowserRouter, Routes,Route } from "react-router-dom";
import { AuthContextProvider } from "./context/AuthContext";
import LoginPage from "./pages/LoginPage";
import { useContext } from "react";
import { AuthContext } from "./context/AuthContext";
import HomePage from "./pages/Home";
import { Navigate } from "react-router-dom";

function RouterD(){
  const context =useContext(AuthContext);
  if (!context) {
    return <div>Error: AuthContext not available</div>;
  }
  const { loggedIn } = context;

    return(
      <BrowserRouter>
        <AuthContextProvider>
        <Routes>
          <Route element={!loggedIn ? <LoginPage /> : <Navigate to="/"/>} path="/login"> </Route>
          <Route element={loggedIn ? <HomePage /> : <Navigate to="/login"/>} path="/"> </Route>
        </Routes>
        </AuthContextProvider>
      </BrowserRouter>
    )
}

export default RouterD;