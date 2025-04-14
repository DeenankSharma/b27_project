import React, { useState, useEffect } from "react";
import { createClient } from '@supabase/supabase-js';
import axios from "axios";
import { serverUrl } from '../utils/api';
import { useNavigate } from "react-router-dom";
import { 
  Box, TextField, Button, Typography, LinearProgress, Paper,
  Container, Grid, Fade, Grow, IconButton, Tooltip,
  CircularProgress, Alert, Chip, Card, CardContent,
  Step, StepLabel, Stepper, Accordion, AccordionSummary, AccordionDetails,
} from '@mui/material';
import { motion } from "framer-motion";
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import HomeIcon from '@mui/icons-material/Home';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import SecurityIcon from '@mui/icons-material/Security';
import MovieIcon from '@mui/icons-material/Movie';
import AudiotrackIcon from '@mui/icons-material/Audiotrack';
import FaceIcon from '@mui/icons-material/Face';
import VisibilityIcon from '@mui/icons-material/Visibility';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { Report, Shorts_Report, VideoTamperingDetectionReport } from "../type";
import './VideoCheck.css';


const supabase = createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_ANON_KEY);

const VideoCheck: React.FC = () => {
  // State variables
  const [wasVidChecked, setWasVidChecked] = useState<Boolean>(false);
  const navigate = useNavigate();
  const [title, setTitle] = useState<string>("");
  const [json_report, setReport] = useState<Report|null>(null);
  const [json_report_for_shorts, setReport_for_shorts] = useState<Shorts_Report|null>(null);
  const [video_tampering_report, setVideoTamperingReport] = useState<VideoTamperingDetectionReport|null>(null);
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [isSubmitValid, setIsSubmitValid] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [pageLoaded, setPageLoaded] = useState<boolean>(false);
  const [activeStep, setActiveStep] = useState<number>(0);
  const [showHelp, setShowHelp] = useState<boolean>(false);

  // Steps for the stepper
  const steps = ['Upload Video', 'Process Video', 'View Results'];

  // Effects
  useEffect(() => {
    setPageLoaded(true);
  }, []);

  useEffect(() => {
    if (selectedFile) {
      setActiveStep(1);
    }
    if (isSubmitValid) {
      setActiveStep(2);
    }
    if (json_report || json_report_for_shorts) {
      setActiveStep(3);
    }
  }, [selectedFile, isSubmitValid, json_report, json_report_for_shorts]);

  // Event handlers
  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTitle(e.target.value);
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  }

  const uploadFile = async () => {
    if (!selectedFile) return;
    if (!title.trim()) {
      setError("Please enter a title for your video");
      return;
    }

    setIsUploading(true);
    setError(null);

    const fileExt = selectedFile.name.split('.').pop();
    const fileName = `${Math.random()}.${fileExt}`;
    const filePath = `${fileName}`;

    try {
      const { data, error: uploadError } = await supabase.storage
        .from('videos-to-check')
        .upload(filePath, selectedFile, {
          cacheControl: '3600',
          upsert: false
        });

      if (uploadError) throw uploadError;
      
      const { data: { publicUrl } } = supabase.storage
        .from('videos-to-check')
        .getPublicUrl(filePath);
      
      setVideoUrl(publicUrl);
      setIsSubmitValid(true);
      setProgress(100);
    } catch (error) {
      console.error('Error uploading file:', error);
      setError("Failed to upload file. Please try again.");
    } finally {
      setIsUploading(false);
    }
  }

  const handleFinalSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsChecking(true);
    setError(null);
    
    try {
      const storeStruct = {
        name: title,
        videoUrl: videoUrl,
      }
      
      const response = await axios.post(`${serverUrl}/test_video_url`, storeStruct);
      
      if (response.status === 200) {
        setWasVidChecked(true);
        
        if (response.data.is_video) {
          const json_report = response.data.message;
          const video_tampering_report = JSON.parse(json_report['Tampering detection result'].replace(/```json|```/g, '').trim());
          setReport(json_report);
          setVideoTamperingReport(video_tampering_report);
        } else if (!response.data.is_video) {
          const shorts_report = response.data.message;
          const video_tampering_report = JSON.parse(shorts_report['Tampering detection result'].replace(/```json|```/g, '').trim());
          setReport_for_shorts(shorts_report);
          setVideoTamperingReport(video_tampering_report);
        }
      }
    } catch (error) {
      console.error('Error checking video:', error);
      setError("Failed to analyze video. Please try again.");
    } finally {
      setIsChecking(false);
    }
  }

  const handleBackToHome = async () => {
    if (wasVidChecked) {
      try {
        const deleteThisVideo = {
          name: title
        }
        await axios.post(`${serverUrl}/delete_video`, deleteThisVideo);
      } catch (error) {
        console.error('Error deleting video:', error);
      }
    }
    navigate('/');
  }

  // Helper functions
  const getVerificationStatusColor = (status: string) => {
    if (status.toLowerCase().includes('verified') || status.toLowerCase().includes('authentic')) {
      return '#4CAF50'; // Green
    } else if (status.toLowerCase().includes('tampered') || status.toLowerCase().includes('manipulated')) {
      return '#F44336'; // Red
    } else {
      return '#FFC107'; // Yellow/Warning
    }
  }

  const getVerificationIcon = (status: string) => {
    if (status.toLowerCase().includes('verified') || status.toLowerCase().includes('authentic')) {
      return <VerifiedUserIcon />;
    } else if (status.toLowerCase().includes('tampered') || status.toLowerCase().includes('manipulated')) {
      return <WarningIcon />;
    } else {
      return <SecurityIcon />;
    }
  }

  const getTamperingScore = () => {
    if (!video_tampering_report) return 0;
    
    const reasons = video_tampering_report['Video Tampering Detection Report']["4. Tampering Detection Summary"]["Reasons for Potential Tampering Detection"].length;
    const isTampered = video_tampering_report['Video Tampering Detection Report']["4. Tampering Detection Summary"]["Tampering Detected"] === "Yes";
    
    return isTampered ? Math.min(100, 50 + (reasons * 10)) : Math.max(0, 40 - (reasons * 10));
  }

  const renderTrustScore = () => {
    const score = 100 - getTamperingScore();
    let color = '#4CAF50';
    let label = 'High Trust';
    
    if (score < 40) {
      color = '#F44336';
      label = 'Low Trust';
    } else if (score < 70) {
      color = '#FFC107';
      label = 'Medium Trust';
    }
    
    return (
      <Box className="trust-score-container">
        <Typography variant="h6" sx={{ mb: 1 }}>Trust Score</Typography>
        <Box className="score-circle">
          <CircularProgress 
            variant="determinate" 
            value={100} 
            size={120} 
            sx={{ 
              color: 'rgba(255, 255, 255, 0.1)',
              position: 'absolute',
            }} 
          />
          <CircularProgress 
            variant="determinate" 
            value={score} 
            size={120} 
            sx={{ 
              color: color,
              position: 'absolute',
            }} 
          />
          <Typography className="score-value" sx={{ color }}>
            {Math.round(score)}%
          </Typography>
        </Box>
        <Typography className="score-label" sx={{ color }}>
          {label}
        </Typography>
      </Box>
    );
  }

  // Help content component
  const helpContent = (
    <Paper className="help-paper">
      <Typography variant="h6" className="help-title">
        How Video Authentication Works
      </Typography>
      
      <Box className="help-feature">
        <Typography variant="subtitle2" className="help-feature-title">
          1. Digital Signature Verification
        </Typography>
        <Typography variant="body2">
          Checks if the video's digital signature matches its content to detect tampering.
        </Typography>
      </Box>
      
      <Box className="help-feature">
        <Typography variant="subtitle2" className="help-feature-title">
          2. Audio Analysis
        </Typography>
        <Typography variant="body2">
          Examines audio for inconsistencies, splices, or artificial generation.
        </Typography>
      </Box>
      
      <Box className="help-feature">
        <Typography variant="subtitle2" className="help-feature-title">
          3. Frame Analysis
        </Typography>
        <Typography variant="body2">
          Analyzes video frames for signs of editing, splicing, or deepfake artifacts.
        </Typography>
      </Box>
      
      <Box className="help-feature">
        <Typography variant="subtitle2" className="help-feature-title">
          4. Object & Face Tracking
        </Typography>
        <Typography variant="body2">
          Detects objects or faces that appear/disappear unnaturally, suggesting manipulation.
        </Typography>
      </Box>
      
      <Button 
        variant="outlined" 
        size="small"
        onClick={() => setShowHelp(false)}
        className="close-help-button"
      >
        Close Help
      </Button>
    </Paper>
  );

  // Main render
  return (
    <Fade in={pageLoaded} timeout={800}>
      <Container maxWidth="lg" className="video-check-container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Box className="page-header">
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <IconButton 
                onClick={handleBackToHome}
                className="back-button"
              >
                <ArrowBackIcon sx={{ color: '#00ED64' }} />
              </IconButton>
              <Typography variant="h4" className="page-title" sx={{ fontSize: { xs: '1.5rem', sm: '2rem', md: '2.125rem' } }}>
                Video Authentication
              </Typography>
            </Box>
            
            <Tooltip title="How it works">
              <IconButton 
                onClick={() => setShowHelp(!showHelp)}
                className="help-button"
              >
                <HelpOutlineIcon sx={{ color: '#00ED64' }} />
              </IconButton>
            </Tooltip>
          </Box>
          <Box sx={{ mb: 3 }}>
            <Stepper activeStep={activeStep} alternativeLabel className="custom-stepper">
              {steps.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </Box>
          
          {showHelp && helpContent}
        </motion.div>

        <Grid container spacing={3}>
          {/* Upload Panel */}
          <Grid item xs={12} md={6}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Box className="panel-container">
                <div className="floating-element-top-right"></div>
                <div className="floating-element-bottom-left"></div>

                <Typography variant="h5" className="section-title">
                  Upload Your Video
                </Typography>
                
                <Typography variant="body2" className="section-description">
                  Our advanced AI system will analyze your video for signs of tampering or manipulation.
                  Upload your video file below to begin the verification process.
                </Typography>

                {error && (
                  <Alert severity="error" sx={{ mb: 3, backgroundColor: 'rgba(211, 47, 47, 0.1)', color: '#ff8a80' }}>
                    {error}
                  </Alert>
                )}

                <TextField
                  fullWidth
                  label="Video Title"
                  variant="outlined"
                  value={title}
                  onChange={handleTitleChange}
                  className="custom-text-field"
                  sx={{ mb: 3, position: 'relative', zIndex: 1 }}
                />
                
                <Box sx={{ mb: 3, position: 'relative', zIndex: 1 }}>
                  <input
                    accept="video/*"
                    style={{ display: 'none' }}
                    id="raised-button-file"
                    type="file"
                    onChange={handleFileInput}
                  />
                  <label htmlFor="raised-button-file">
                    <Button
                      variant="outlined"
                      component="span"
                      startIcon={<CloudUploadIcon />}
                      fullWidth
                      className="upload-button"
                    >
                      Select Video File
                    </Button>
                  </label>
                </Box>
                
                {selectedFile && (
                  <Grow in={!!selectedFile} timeout={500}>
                    <Paper elevation={0} className="file-info-paper">
                      <Typography variant="body2">
                        <b>Selected file:</b> {selectedFile.name}
                      </Typography>
                      <Typography variant="body2">
                        <b>Size:</b> {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                      </Typography>
                      <Typography variant="body2">
                        <b>Type:</b> {selectedFile.type || "Unknown"}
                      </Typography>
                      <Typography variant="body2">
                        <b>Last modified:</b> {new Date(selectedFile.lastModified).toLocaleString()}
                      </Typography>
                    </Paper>
                  </Grow>
                )}
                
                <Button
                  onClick={uploadFile}
                  variant="contained"
                  disabled={!selectedFile || isUploading}
                  startIcon={isUploading ? <CircularProgress size={20} color="inherit" /> : <CloudUploadIcon />}
                  className="submit-button"
                  sx={{ position: 'relative', zIndex: 1 }}
                >
                  {isUploading ? 'Uploading...' : 'Upload Video'}
                </Button>
                
                {progress > 0 && (
                  <Box className="progress-container">
                    <Typography variant="body2" sx={{ color: 'white', mb: 1 }}>
                      Upload Progress: {progress}%
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={progress} 
                      className="custom-progress"
                    />
                  </Box>
                )}
                
                <Button
                  onClick={handleFinalSubmit}
                  variant="contained"
                  disabled={!isSubmitValid || isChecking}
                  startIcon={isChecking ? <CircularProgress size={20} color="inherit" /> : <CheckCircleOutlineIcon />}
                  className="submit-button"
                  sx={{ position: 'relative', zIndex: 1, mt: 'auto' }}
                >
                  {isChecking ? 'Analyzing...' : 'Verify Video Authenticity'}
                </Button>
                
                {isChecking && (
                  <Box sx={{ mt: 3, textAlign: 'center' }}>
                    <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 1 }}>
                      Analyzing video for tampering...
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                      {['Signature', 'Audio', 'Frames', 'Objects', 'Faces'].map((step, index) => (
                        <Chip 
                          key={index}
                          label={step}
                          size="small"
                          className="analysis-chip"
                          sx={{ animation: `pulse 1.5s infinite ${index * 0.3}s` }}
                        />
                      ))}
                    </Box>
                  </Box>
                )}
              </Box>
            </motion.div>
          </Grid>

          {/* Results Panel */}
          <Grid item xs={12} md={6}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <Box className="panel-container">
                <div className="floating-element-top-left"></div>
                
                <Typography variant="h5" className="section-title">
                  Authentication Report
                </Typography>

                {!json_report && !json_report_for_shorts ? (
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    height: '100%',
                    textAlign: 'center',
                    color: 'rgba(255, 255, 255, 0.7)',
                    position: 'relative',
                    zIndex: 1
                  }}>
                                        <SecurityIcon sx={{ fontSize: 60, color: '#00ED64', opacity: 0.5, mb: 2 }} />
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      No Report Available
                    </Typography>
                    <Typography variant="body2">
                      Upload and verify a video to see the authentication report here.
                    </Typography>
                  </Box>
                ) : (
                  <Box sx={{ position: 'relative', zIndex: 1 }}>
                    {renderTrustScore()}
                    
                    <Card className="info-card">
                      <CardContent>
                        <Typography variant="h6" className="info-card-title">
                          Verification Results
                        </Typography>
                        
                        {json_report && (
                          <>
                            <Box className="feature-item">
                              <SecurityIcon className="feature-icon" />
                              <Typography variant="body2">
                                <b>Signature:</b> {json_report["Signature verification result"]}
                              </Typography>
                            </Box>
                            
                            <Box className="feature-item">
                              <MovieIcon className="feature-icon" />
                              <Typography variant="body2">
                                <b>Tampering:</b> {video_tampering_report?.["Video Tampering Detection Report"]["4. Tampering Detection Summary"]["Tampering Detected"]}
                              </Typography>
                            </Box>
                            
                            <Box className="feature-item">
                              <AudiotrackIcon className="feature-icon" />
                              <Typography variant="body2">
                                <b>Audio:</b> {json_report["Audio analysis result"]}
                              </Typography>
                            </Box>
                            
                            <Box className="feature-item">
                              <FaceIcon className="feature-icon" />
                              <Typography variant="body2">
                                <b>Deepfake Probability:</b> {json_report["deepfake chances"]}%
                              </Typography>
                            </Box>
                          </>
                        )}
                        
                        {json_report_for_shorts && (
                          <>
                            <Box className="feature-item">
                              <SecurityIcon className="feature-icon" />
                              <Typography variant="body2">
                                <b>Signature:</b> {json_report_for_shorts["Signature verification result"]}
                              </Typography>
                            </Box>
                            
                            <Box className="feature-item">
                              <MovieIcon className="feature-icon" />
                              <Typography variant="body2">
                                <b>Tampering:</b> {video_tampering_report?.["Video Tampering Detection Report"]["4. Tampering Detection Summary"]["Tampering Detected"]}
                              </Typography>
                            </Box>
                            
                            <Box className="feature-item">
                              <FaceIcon className="feature-icon" />
                              <Typography variant="body2">
                                <b>Deepfake Probability:</b> {json_report_for_shorts["deepfake chances"]}%
                              </Typography>
                            </Box>
                          </>
                        )}
                      </CardContent>
                    </Card>
                    
                    {video_tampering_report && (
                      <Box sx={{ mt: 3 }}>
                        <Typography variant="h6" sx={{ mb: 2, color: 'white' }}>
                          Detailed Analysis
                        </Typography>
                        
                        <Accordion className="report-accordion">
                          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#00ED64' }} />}>
                            <Typography>Shot Change Analysis</Typography>
                          </AccordionSummary>
                          <AccordionDetails className="accordion-content">
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Average Shot Duration:
                              </Typography>
                              <Typography variant="body2" className="report-detail-value">
                                {video_tampering_report["Video Tampering Detection Report"]["1. Shot Change Analysis"]["Average Shot Duration"]}
                              </Typography>
                            </Box>
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Rapid Shot Changes:
                              </Typography>
                              <Typography variant="body2" className="report-detail-value">
                                {video_tampering_report["Video Tampering Detection Report"]["1. Shot Change Analysis"]["Number of Rapid Shot Changes"]}
                              </Typography>
                            </Box>
                          </AccordionDetails>
                        </Accordion>
                        
                        <Accordion className="report-accordion">
                          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#00ED64' }} />}>
                            <Typography>Motion Analysis</Typography>
                          </AccordionSummary>
                          <AccordionDetails className="accordion-content">
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Motion Consistency:
                              </Typography>
                              <Typography 
                                variant="body2" 
                                // className={`report-detail-value ${
                                //   video_tampering_report["Video Tampering Detection Report"]["2. Motion Analysis"]["Motion Consistency"] === "High" 
                                //     ? "positive" 
                                //     : video_tampering_report["Video Tampering Detection Report"]["2. Motion Analysis"]["Motion Consistency"] === "Low" 
                                //       ? "negative" 
                                //       : "warning"
                                // }`}
                              >
                                {video_tampering_report["Video Tampering Detection Report"]["2. Motion Analysis"]["Motion Consistency"]}
                              </Typography>
                            </Box>
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Unnatural Motion Detected:
                              </Typography>
                              <Typography 
                                variant="body2" 
                                className={`report-detail-value ${
                                  video_tampering_report["Video Tampering Detection Report"]["2. Motion Analysis"]["Unnatural Motion Detected"] === "No" 
                                    ? "positive" 
                                    : "negative"
                                }`}
                              >
                                {video_tampering_report["Video Tampering Detection Report"]["2. Motion Analysis"]["Unnatural Motion Detected"]}
                              </Typography>
                            </Box>
                          </AccordionDetails>
                        </Accordion>
                        
                        <Accordion className="report-accordion">
                          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#00ED64' }} />}>
                            <Typography>Visual Artifacts</Typography>
                          </AccordionSummary>
                          <AccordionDetails className="accordion-content">
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Compression Artifacts:
                              </Typography>
                              <Typography 
                                variant="body2" 
                                className={`report-detail-value ${
                                  video_tampering_report["Video Tampering Detection Report"]["3. Visual Artifacts Analysis"]["Compression Artifacts"] === "Low" 
                                    ? "positive" 
                                    : video_tampering_report["Video Tampering Detection Report"]["3. Visual Artifacts Analysis"]["Compression Artifacts"] === "High" 
                                      ? "negative" 
                                      : "warning"
                                }`}
                              >
                                {video_tampering_report["Video Tampering Detection Report"]["3. Visual Artifacts Analysis"]["Compression Artifacts"]}
                              </Typography>
                            </Box>
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Boundary Artifacts:
                              </Typography>
                              <Typography 
                                variant="body2" 
                                className={`report-detail-value ${
                                  video_tampering_report["Video Tampering Detection Report"]["3. Visual Artifacts Analysis"]["Boundary Artifacts"] === "Not Detected" 
                                    ? "positive" 
                                    : "negative"
                                }`}
                              >
                                {video_tampering_report["Video Tampering Detection Report"]["3. Visual Artifacts Analysis"]["Boundary Artifacts"]}
                              </Typography>
                            </Box>
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Inconsistent Lighting:
                              </Typography>
                              <Typography 
                                variant="body2" 
                                className={`report-detail-value ${
                                  video_tampering_report["Video Tampering Detection Report"]["3. Visual Artifacts Analysis"]["Inconsistent Lighting"] === "Not Detected" 
                                    ? "positive" 
                                    : "negative"
                                }`}
                              >
                                {video_tampering_report["Video Tampering Detection Report"]["3. Visual Artifacts Analysis"]["Inconsistent Lighting"]}
                              </Typography>
                            </Box>
                          </AccordionDetails>
                        </Accordion>
                        
                        <Accordion className="report-accordion">
                          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#00ED64' }} />}>
                            <Typography>Tampering Summary</Typography>
                          </AccordionSummary>
                          <AccordionDetails className="accordion-content">
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Tampering Detected:
                              </Typography>
                              <Typography 
                                variant="body2" 
                                className={`report-detail-value ${
                                  video_tampering_report["Video Tampering Detection Report"]["4. Tampering Detection Summary"]["Tampering Detected"] === "No" 
                                    ? "positive" 
                                    : "negative"
                                }`}
                                sx={{ fontWeight: 'bold' }}
                              >
                                {video_tampering_report["Video Tampering Detection Report"]["4. Tampering Detection Summary"]["Tampering Detected"]}
                              </Typography>
                            </Box>
                            <Box className="report-detail-item">
                              <Typography variant="body2" className="report-detail-label">
                                Confidence Level:
                              </Typography>
                              <Typography 
                                variant="body2" 
                                className={`report-detail-value ${
                                  video_tampering_report["Video Tampering Detection Report"]["4. Tampering Detection Summary"]["Tampering Detected"] === "No" 
                                    ? "positive" 
                                    : "negative"
                                }`}
                              >
                                {video_tampering_report["Video Tampering Detection Report"]["4. Tampering Detection Summary"]["Confidence Level"]}
                              </Typography>
                            </Box>
                            
                            <Typography variant="body2" sx={{ mb: 1, mt: 2, fontWeight: 'bold' }}>
                              Reasons for Detection:
                            </Typography>
                            
                            {video_tampering_report["Video Tampering Detection Report"]["4. Tampering Detection Summary"]["Reasons for Potential Tampering Detection"].map((reason: string, index: number) => (
                              <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                                <div className="reason-bullet"></div>
                                <Typography variant="body2">{reason}</Typography>
                              </Box>
                            ))}
                          </AccordionDetails>
                        </Accordion>
                      </Box>
                    )}
                    
                    <Box className="info-box">
                      <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center' }}>
                        <InfoIcon sx={{ fontSize: 16, mr: 1, color: '#00ED64' }} />
                        This report is based on AI analysis and should be used as a guide. For critical verification, consult with digital forensics experts.
                      </Typography>
                    </Box>
                  </Box>
                )}
              </Box>
            </motion.div>
          </Grid>
        </Grid>
      </Container>
    </Fade>
  );
};

export default VideoCheck;
