import React from 'react';
import { useAuth } from '../context/AuthContext';
import UploadVideoToS3WithNativeSdk from '@/components/VideoUpload';
import VideosFetch from '@/components/VideosFetch';
import StyledDrawer from '@/components/SideBar';
import { useSidebarContext } from '@/context/TabContext';
import { Box, Typography, Container, Grid } from '@mui/material';

const Home: React.FC = () => {
  const { loggedIn, user } = useAuth();
  const { selectedItem } = useSidebarContext();

  if (loggedIn === true) {
    return (
      <Box sx={{ 
        display: 'flex', 
        minHeight: '100vh',
        background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,20,0.9) 100%)',
      }}>
        <StyledDrawer />
        
        <Box component="main" sx={{ 
          flexGrow: 1, 
          p: 3, 
          ml: '250px',
          display: 'flex',
          flexDirection: 'column'
        }}>
          {/* Header Section */}
          <Box sx={{ 
            mb: 4, 
            mt: 2,
            display: 'flex',
            flexDirection: 'column'
          }}>
            <Typography 
              variant="h4" 
              sx={{ 
                color: 'white',
                fontFamily: 'Space Grotesk, sans-serif',
                fontWeight: 700,
                mb: 1
              }}
            >
              {selectedItem === 0 ? 'Upload New Video' : 'Your Video Library'}
            </Typography>
            
            <Typography 
              variant="body1" 
              sx={{ 
                color: 'rgba(255,255,255,0.7)',
                fontFamily: 'Space Grotesk, sans-serif',
                mb: 2
              }}
            >
              {selectedItem === 0 
                ? 'Share your content securely with VidTrust technology' 
                : 'Manage and view your uploaded videos'}
            </Typography>
            
            {/* Decorative line */}
            <Box sx={{ 
              width: '100px', 
              height: '4px', 
              background: '#00ED64',
              borderRadius: '2px',
              mb: 3
            }} />
          </Box>

          {/* Content Section */}
          <Box sx={{ 
            flexGrow: 1,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'flex-start',
            width: '100%'
          }}>
            {selectedItem === 0 ? <UploadVideoToS3WithNativeSdk /> : <VideosFetch />}
          </Box>
        </Box>
      </Box>
    );
  }

  return null;
};

export default Home;