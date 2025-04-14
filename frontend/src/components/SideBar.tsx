import * as React from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import DriveFolderUploadIcon from '@mui/icons-material/DriveFolderUpload';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import Dashboard from './Dashboard';
import { useSidebarContext } from '../context/TabContext.tsx';

export default function StyledDrawer() {
  const [open, setOpen] = React.useState(true);
  const { selectedItem, updateSelectedItem } = useSidebarContext();
  
  const DrawerList = (
    <Box 
      sx={{ 
        width: 250, 
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.4)',
        backdropFilter: 'blur(10px)',
        boxShadow: '0 0 20px 5px rgba(0, 237, 100, 0.2)',
        display: 'flex',
        flexDirection: 'column',
        borderRight: '1px solid rgba(0, 237, 100, 0.2)',
      }} 
      role="presentation"
    >
      <Box sx={{ 
        padding: '30px 0 20px',
        borderBottom: '1px solid rgba(0, 237, 100, 0.1)',
      }}>
        <h2 style={{
          color: 'white',
          textAlign: 'center',
          fontSize: "2.5rem",
          fontFamily: "Space Grotesk, sans-serif",
          fontWeight: "700",
          margin: 0,
          letterSpacing: '-0.5px',
        }}>
          <span style={{ color:"#00ED64" }}>Vid</span>Trust
        </h2>
      </Box>
      
      <List sx={{ 
        flexGrow: 1, 
        padding: '20px 0',
      }}>
        {[
          { text: 'Upload Videos', icon: <DriveFolderUploadIcon /> },
          { text: 'Existing Videos', icon: <VideoLibraryIcon /> }
        ].map((item, index) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 1 }}>
            <ListItemButton 
              onClick={() => updateSelectedItem(index)}
              sx={{
                margin: '0 10px',
                borderRadius: '8px',
                padding: '10px 16px',
                backgroundColor: selectedItem === index ? 'rgba(0, 237, 100, 0.15)' : 'transparent',
                transition: 'all 0.2s ease',
                '&:hover': {
                  backgroundColor: 'rgba(0, 237, 100, 0.1)',
                  transform: 'translateX(5px)',
                },
              }}
            >
              <ListItemIcon sx={{ 
                color: selectedItem === index ? '#00ED64' : 'rgba(0, 237, 100, 0.7)',
                minWidth: '40px',
              }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text} 
                sx={{ 
                  '& .MuiListItemText-primary': {
                    color: 'white',
                    fontFamily: 'Space Grotesk, sans-serif',
                    fontWeight: 500,
                    fontSize: '1rem',
                  }
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      <Box sx={{ p: 2 }}>
        <Dashboard />
      </Box>
    </Box>
  );

  return (
    <div>
      <Drawer 
        open={open}
        variant="persistent"
        PaperProps={{
          sx: {
            backgroundColor: 'transparent',
            boxShadow: 'none',
            width: 250,
          }
        }}
      >
        {DrawerList}
      </Drawer>
    </div>
  );
}