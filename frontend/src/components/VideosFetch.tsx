import React, { useState, useEffect } from "react";
import axios from "axios";
import { serverUrl } from '../utils/api';

interface VideoUrl {
  id: string;
  email: string;
  title: string;
  imageUrl: string;
  videoUrl: string;
  date:string;
}

const VideosFetch: React.FC = () => {
  const [urls, setUrls] = useState<VideoUrl[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoUrl | null>(null);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);

  const get_video_urls = async () => {
    try {
      const response = await axios.get<VideoUrl[]>(`${serverUrl}/auth/video_fetch_url`);
      setUrls(response.data);
    } catch (err) {
      console.error("Error fetching video URLs:", err);
    }
  };
  // useEffect(()=>{
  //   setUrls([
  //     {
  //       id: "1",
  //       email: "user1@example.com",
  //       title: "Beautiful Sunset Time Lapse",
  //       imageUrl: "https://images.unsplash.com/photo-1616832880699-8541b04005ec?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8c3Vuc2V0JTIwdGltZWxhcHNlfGVufDB8fDB8fHww",
  //       videoUrl: "https://player.vimeo.com/external/314181352.sd.mp4?s=9f9d84c323ec55b23e2858e3452d119f1679beaa&profile_id=164&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "2",
  //       email: "user2@example.com",
  //       title: "Aerial View of Beach",
  //       imageUrl: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8YmVhY2h8ZW58MHx8MHx8fDA%3D",
  //       videoUrl: "https://player.vimeo.com/external/434045526.sd.mp4?s=c27eecc69a27dbc4ff2b87d38afc35f1a9e7c02d&profile_id=164&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "3",
  //       email: "user3@example.com",
  //       title: "City Traffic at Night",
  //       imageUrl: "https://images.unsplash.com/photo-1506868544459-7f2e136d1d91?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nnx8Y2l0eSUyMG5pZ2h0fGVufDB8fDB8fHww",
  //       videoUrl: "https://player.vimeo.com/external/342571552.hd.mp4?s=6aa6f164de3812abadff3dde86d19f7a074a8a66&profile_id=175&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "4",
  //       email: "user4@example.com",
  //       title: "Waves Crashing on Rocky Shore",
  //       imageUrl: "https://images.unsplash.com/photo-1619806640513-a68a83a8614e?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8d2F2ZXMlMjBjcmFzaGluZ3xlbnwwfHwwfHx8MA%3D%3D",
  //       videoUrl: "https://player.vimeo.com/external/332588783.hd.mp4?s=a3620eb7ff73a2bef5e8fae7a451514ad1305e49&profile_id=175&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "5",
  //       email: "user5@example.com",
  //       title: "Vibrant Northern Lights",
  //       imageUrl: "https://images.unsplash.com/photo-1563974318767-a4de855d7b43?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8bm9ydGhlcm4lMjBsaWdodHN8ZW58MHx8MHx8fDA%3D",
  //       videoUrl: "https://player.vimeo.com/external/236075858.hd.mp4?s=539faad12f040eb5afd8de3160db1220f1a35b3d&profile_id=175&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "6",
  //       email: "user6@example.com",
  //       title: "Serene Forest Stream",
  //       imageUrl: "https://images.unsplash.com/photo-1421789665209-c9b2a435e3dc?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Zm9yZXN0JTIwc3RyZWFtfGVufDB8fDB8fHww",
  //       videoUrl: "https://player.vimeo.com/external/189545487.sd.mp4?s=8cd2af1ec08f7ce121a5a6a09c78c05237943524&profile_id=164&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "7",
  //       email: "user5@example.com",
  //       title: "Vibrant Northern Lights",
  //       imageUrl: "https://images.unsplash.com/photo-1563974318767-a4de855d7b43?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8bm9ydGhlcm4lMjBsaWdodHN8ZW58MHx8MHx8fDA%3D",
  //       videoUrl: "https://player.vimeo.com/external/236075858.hd.mp4?s=539faad12f040eb5afd8de3160db1220f1a35b3d&profile_id=175&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "8",
  //       email: "user6@example.com",
  //       title: "Serene Forest Stream",
  //       imageUrl: "https://images.unsplash.com/photo-1421789665209-c9b2a435e3dc?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Zm9yZXN0JTIwc3RyZWFtfGVufDB8fDB8fHww",
  //       videoUrl: "https://player.vimeo.com/external/189545487.sd.mp4?s=8cd2af1ec08f7ce121a5a6a09c78c05237943524&profile_id=164&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "9",
  //       email: "user5@example.com",
  //       title: "Vibrant Northern Lights",
  //       imageUrl: "https://images.unsplash.com/photo-1563974318767-a4de855d7b43?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8bm9ydGhlcm4lMjBsaWdodHN8ZW58MHx8MHx8fDA%3D",
  //       videoUrl: "https://player.vimeo.com/external/236075858.hd.mp4?s=539faad12f040eb5afd8de3160db1220f1a35b3d&profile_id=175&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     },
  //     {
  //       id: "10",
  //       email: "user6@example.com",
  //       title: "Serene Forest Stream",
  //       imageUrl: "https://images.unsplash.com/photo-1421789665209-c9b2a435e3dc?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Zm9yZXN0JTIwc3RyZWFtfGVufDB8fDB8fHww",
  //       videoUrl: "https://player.vimeo.com/external/189545487.sd.mp4?s=8cd2af1ec08f7ce121a5a6a09c78c05237943524&profile_id=164&oauth2_token_id=57447761",
  //       date : '10 - 10 - 2024'
  //     }
  //   ])
  // },[])
  

  useEffect(()=>{
    get_video_urls()
  },[])

  const playVideo = (video: VideoUrl) => {
    setSelectedVideo(video);
    setIsVideoPlaying(true);
  };

  const closeVideo = () => {
    setSelectedVideo(null);
    setIsVideoPlaying(false);
  };

  return (
    <div style={styles.container}>
      {isVideoPlaying && selectedVideo && (
        <div style={styles.overlay}>
          <div style={styles.modalContent}>
            <div style={styles.videoWrapper}>
              <video 
                src={selectedVideo.videoUrl} 
                controls 
                autoPlay 
                style={styles.video}
              >
                Your browser does not support the video tag.
              </video>
            </div>
            <h2 style={styles.modalTitle}>{selectedVideo.title}</h2>
            <button 
              onClick={closeVideo}
              style={styles.closeButton}
            >
              Close
            </button>
          </div>
        </div>
      )}

      <div style={styles.grid}>
        {urls.map((video) => (
          <div key={video.id} style={styles.card}>
            <div style={{...styles.thumbnail, backgroundImage: `url(${video.imageUrl})`}} />
            <div style={styles.cardContent}>
              <h3 style={styles.cardTitle}>{video.title}</h3>
              <button 
                onClick={() => playVideo(video)}
                style={styles.playButton}
              >
                Play
              </button>
              <h3 style={styles.cardTitle}>{video.date}</h3>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const styles = {
  container: {
    position: 'absolute' as const,
    right:0,
    top:0,
    left:250,
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    padding: '40px',
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    borderRadius: '15px',
    boxShadow: '0 0 10px 5px rgba(0, 237, 100, 0.3)',
    backdropFilter: 'blur(5px)',
    maxWidth: '1150px',
    margin: '0 auto',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
    gap: '20px',
    width: '100%',
  },
  card: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '8px',
    overflow: 'hidden',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  thumbnail: {
    width: '100%',
    height: '200px',
    backgroundSize: 'cover',
    backgroundPosition: 'center',
  },
  cardContent: {
    padding: '15px',
  },
  cardTitle: {
    fontSize: '18px',
    fontWeight: 'bold',
    marginBottom: '10px',
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    color: 'white',
    fontFamily: 'Space Grotesk, sans-serif',
  },
  playButton: {
    backgroundColor: '#00ED64',
    color: 'black',
    padding: '8px 16px',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
    transition: 'background-color 0.3s',
  },
  overlay: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  modalContent: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    padding: '20px',
    borderRadius: '8px',
    maxWidth: '80%',
    width: '800px',
    boxShadow: '0 0 10px 5px rgba(0, 237, 100, 0.3)',
  },
  videoWrapper: {
    position: 'relative' as const,
    paddingTop: '56.25%', 
  },
  video: {
    position: 'absolute' as const,
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
  },
  modalTitle: {
    fontSize: '24px',
    fontWeight: 'bold',
    marginTop: '15px',
    marginBottom: '15px',
    color: 'white',
    fontFamily: 'Space Grotesk, sans-serif',
  },
  closeButton: {
    backgroundColor: '#00ED64',
    color: 'black',
    padding: '10px 20px',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold',
    transition: 'background-color 0.3s',
  },
};

export default VideosFetch;