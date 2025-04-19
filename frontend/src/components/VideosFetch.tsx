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
    <div style={{
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'flex-start',
      padding: '0',
      margin: '0',
      boxSizing: 'border-box'
    }}>
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

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
        gap: '20px',
        width: '100%',
        padding: '0',
        margin: '0',
        boxSizing: 'border-box'
      }}>
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
  // Keep the rest of the styles as they are
  overlay: {
    position: 'fixed' as 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
    backdropFilter: 'blur(5px)',
  },
  modalContent: {
    backgroundColor: 'rgba(18, 18, 18, 0.95)',
    borderRadius: '12px',
    padding: '20px',
    maxWidth: '90%',
    maxHeight: '90%',
    display: 'flex',
    flexDirection: 'column' as 'column',
    alignItems: 'center',
    border: '1px solid rgba(0, 237, 100, 0.3)',
    boxShadow: '0 0 20px rgba(0, 237, 100, 0.2)',
  },
  videoWrapper: {
    width: '100%',
    maxWidth: '800px',
    borderRadius: '8px',
    overflow: 'hidden',
    marginBottom: '15px',
  },
  video: {
    width: '100%',
    display: 'block',
  },
  modalTitle: {
    color: 'white',
    margin: '0 0 15px 0',
    fontFamily: 'Space Grotesk, sans-serif',
    fontSize: '1.5rem',
  },
  closeButton: {
    backgroundColor: 'rgba(0, 237, 100, 0.2)',
    color: '#00ED64',
    border: '1px solid #00ED64',
    padding: '8px 16px',
    borderRadius: '4px',
    cursor: 'pointer',
    fontFamily: 'Space Grotesk, sans-serif',
    transition: 'all 0.3s ease',
  },
  card: {
    backgroundColor: 'rgba(18, 18, 18, 0.7)',
    borderRadius: '12px',
    overflow: 'hidden',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
    cursor: 'pointer',
    height: '100%',
    display: 'flex',
    flexDirection: 'column' as 'column',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  thumbnail: {
    width: '100%',
    height: '160px',
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  },
  cardContent: {
    padding: '15px',
    display: 'flex',
    flexDirection: 'column' as 'column',
    flexGrow: 1,
  },
  cardTitle: {
    color: 'white',
    margin: '0 0 10px 0',
    fontFamily: 'Space Grotesk, sans-serif',
    fontSize: '1rem',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as 'nowrap',
  },
  playButton: {
    backgroundColor: 'rgba(0, 237, 100, 0.2)',
    color: '#00ED64',
    border: '1px solid #00ED64',
    padding: '8px 0',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: 'auto',
    marginBottom: '10px',
    fontFamily: 'Space Grotesk, sans-serif',
    transition: 'all 0.3s ease',
  }
};

export default VideosFetch;