#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <algorithm> 
#include <string>

void pixelFrequency(cv::Mat& img, int intensity[])
{
    for (int j = 0; j < img.rows; j++)
        for (int i = 0; i < img.cols; i++)
            intensity[int(img.at<uchar>(j, i))]++;
}
void pixelProbability(cv::Mat& img, double probability[], int intensity[])
{
    for (int i = 0; i < 256; i++)
        probability[i] = intensity[i] / double(img.rows * img.cols);
}
void cumuProbability(double probability[], double cumulativeProbability[])
{
    cumulativeProbability[0] = probability[0];
    for (int i = 1; i < 256; i++)
        cumulativeProbability[i] = probability[i] + cumulativeProbability[i - 1];
}
void histogramEqualization(cv::Mat& img, int intensity[], double probability[], double cumulativeProbability[])
{
    pixelFrequency(img, intensity);
    pixelProbability(img, probability, intensity);
    cumuProbability(probability, cumulativeProbability);
    for (int i = 0; i < 256; i++)
        cumulativeProbability[i] = floor(cumulativeProbability[i] * 255);
    for (int j = 0; j < img.rows; j++)
    {
        for (int i = 0; i < img.cols; i++)
        {
            img.at<uchar>(j, i) = cumulativeProbability[int(img.at<uchar>(j, i))];
        }
    }
}

void showHistogram(cv::Mat& image, std::string fileName = "")
{
    int bins = 256;             // number of bins
    int nc = image.channels();    // number of channels
    std::vector<cv::Mat> histogram(nc);       // array for storing the histograms
    std::vector<cv::Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = {0,0,0};      // peak value for each histogram

    
	for (int i = 0; i < histogram.size(); i++)
    histogram[i] = cv::Mat::zeros(1, bins, CV_32SC1);

	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			for (int k = 0; k < nc; k++){
				uchar val = nc == 1 ? image.at<uchar>(i,j) : image.at<cv::Vec3b>(i,j)[k];
				histogram[k].at<int>(val) += 1;
			}
		}
	}

	for (int i = 0; i < nc; i++){
		for (int j = 0; j < bins-1; j++)
			hmax[i] = histogram[i].at<int>(j) > hmax[i] ? histogram[i].at<int>(j) : hmax[i];
	}

	const char* wname[3] = { "Blue", "Green", "Red" };
	cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0), cv::Scalar(0,0,255) };

	for (int i = 0; i < nc; i++){
		canvas[i] = cv::Mat::ones(125, bins, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < bins-1; j++){
			line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (histogram[i].at<int>(j) * rows/hmax[i])),
				nc == 1 ? cv::Scalar(255, 255, 255) : colors[i],
				1, 8, 0
			);
		}

		cv::imshow(nc == 1 ? fileName : wname[i]+fileName, canvas[i]);
		// string name = string(wname[i])+".jpg";
		// imwrite(nc == 1 ? fileName+".jpg" : name, canvas[i]);
	}
}


void RGB_HSI(cv::Mat img,cv::Mat& intensity,cv::Mat& Hue,cv::Mat& Saturation)
{

	int H = img.rows;
	int W = img.cols;

	for (int j=0;j<H;j++)
	{ 
	 for (int i=0;i<W;i++) 
	 {

	     double temp = 0;
	     double R =(double) img.at<cv::Vec3b>(j,i).val[2];
	     double G =(double) img.at<cv::Vec3b>(j,i).val[1]; 
	     double B =(double) img.at<cv::Vec3b>(j,i).val[0];
	     double intensity_ = 0;
	     double hue_ = 0;
	     double saturation_ = 0;

	   intensity_ = (R + G + B) / 3;

	   if ((R + G + B) == 765) {
	      saturation_ = 0;
	      hue_ = 0;
	        }

	  double minimum = cv::min(R, cv::min(G, B));

	  if (intensity_ > 0) {
	   saturation_ = 1 - minimum / intensity_;
	   }
	  else if (intensity_ == 0) {
	   saturation_ = 0;
	   }            

	  if(saturation_ != 0)
	  {
	  	temp = (R - (G/2) - (B/2)) / (sqrt((R*R) + (G*G) + (B*B) - (R*G) - (R*B) - (G*B)));
		if (G >= B) {
		    hue_ = acos(temp); 
		}

	    else if (B > G) {     
	    hue_ = (360* (3.14159265)) / 180.0 - acos(temp);
	    }
	  }	



	  intensity.at<uchar>(j,i) = (int) intensity_;
	  Saturation.at<double>(j,i) = saturation_;	
	  Hue.at<uchar>(j,i) = (int) ((hue_*180)/3.14159265);

	}
	
	}
	
}

void HSI_RGB(cv::Mat& img,cv::Mat& intensity,cv::Mat& Hue,cv::Mat& Saturation)
{
	int resultHue = 0;
	double resultSaturation = 0;
	int resultIntensity = 0;

	for(int j = 0;j < img.rows;++j)
	{
		for(int i = 0;i < img.cols;++i)
		{
			int backR = 0, backG = 0, backB = 0;
			resultHue = Hue.at<uchar>(j,i);
			resultIntensity = intensity.at<uchar>(j,i);
			resultSaturation = Saturation.at<double>(j,i);

			if (resultHue == 0){
			   backR = (int) (resultIntensity + (2 * resultIntensity * resultSaturation));
			   backG = (int) (resultIntensity - (resultIntensity * resultSaturation));
			   backB = (int) (resultIntensity - (resultIntensity * resultSaturation));
			  }

			else if ((0 < resultHue) && (resultHue < 120)) {	
			   backR = (int) resultIntensity*(1 + (resultSaturation*cos(resultHue*3.14159265/180.0)/cos((60 - resultHue)*3.14159265/180.0)));
			   backB = (int) (resultIntensity - (resultIntensity * resultSaturation));
			   backG = (int) (3*resultIntensity - (backB+backR));
			  }

			else if ( resultHue == 120 ){
			   backR = (int) (resultIntensity - (resultIntensity * resultSaturation));
			   backG = (int) (resultIntensity + (2 * resultIntensity * resultSaturation));
			   backB = (int) (resultIntensity - (resultIntensity * resultSaturation));
			  }

			else if ((120 < resultHue) && (resultHue < 240)) {
			   resultHue = resultHue - 120;
			  
               backG = resultIntensity*(1 + (resultSaturation*cos(resultHue*3.14159265/180.0)/cos((60 - resultHue)*3.14159265/180.0)));	
			   backR = (int) (resultIntensity - (resultIntensity * resultSaturation));
			   backB = (int) (3*resultIntensity - (backG+backR));			  }

			else if (resultHue == 240) {
			   backR = (int) (resultIntensity - (resultIntensity * resultSaturation));
			   backG = (int) (resultIntensity - (resultIntensity * resultSaturation));
			   backB = (int) (resultIntensity + (2 * resultIntensity * resultSaturation));
			  }

			else if ((240 < resultHue) && (resultHue < 360)) {
			   backR = (int) (resultIntensity + (resultIntensity * resultSaturation) * (1 - cos((resultHue-240)*3.14159265/180.0) / cos((300-resultHue)*3.14159265/180.0)));
			   backG = (int) (resultIntensity - (resultIntensity * resultSaturation));
			   backB = (int) (resultIntensity + (resultIntensity * resultSaturation) * cos((resultHue-240)*3.14159265/180.0) / cos((300-resultHue)*3.14159265/180.0));
			  }
			
			  if(backR < 0)
                {backR = 0;}
              if(backR > 255)
				{backR = 255;}
              if(backB < 0)
                {backB = 0;}
              if(backB > 255)
				{backB = 255;}
              if(backG < 0)
                {backG = 0;}
              if(backG > 255)
				{backG = 255;}


			img.at<cv::Vec3b>(j, i)[0] = backB;
            img.at<cv::Vec3b>(j, i)[1] = backG;
			img.at<cv::Vec3b>(j, i)[2] = backR;

		}
	}

}


void histequalize(char* address)
{
	cv::Mat img = cv::imread(address,CV_LOAD_IMAGE_UNCHANGED);

	int intensity[256] = { 0 };
    double probabilityimg[256] = { 0 };
    double cumulativeProbability[256] = { 0 };

	if(img.channels() == 1)
	{
		cv::Mat equalizedimg = cv::imread(address, CV_LOAD_IMAGE_GRAYSCALE);
	    histogramEqualization(equalizedimg, intensity, probabilityimg, cumulativeProbability);
	    
	    cv::namedWindow("Originalimage", cv::WINDOW_AUTOSIZE);
	    cv::imshow("Originalimage", img);
	    
	    cv::namedWindow("equalizedimage", cv::WINDOW_AUTOSIZE);
	    cv::imshow("equalizedimage", equalizedimg);
	    cv::imwrite( "/home/pritish/Downloads/equalizedimg.jpg", equalizedimg );

	    showHistogram(img,"original");
	    showHistogram(equalizedimg,"equalizedimg");
	    
	    return;	
	}

	if(img.channels() >= 3)
	{
		cv::Mat Intensity = cv::Mat::zeros(img.rows,img.cols, CV_8UC1);
		cv::Mat Hue = cv::Mat::zeros(img.rows,img.cols, CV_8UC1);
		cv::Mat Saturation = cv::Mat::zeros(cv::Size(img.rows,img.cols), CV_64FC1);

		cv::Mat finalimg = cv::Mat::zeros(img.rows,img.cols, CV_8UC3);

		RGB_HSI(img,Intensity,Hue,Saturation);

		cv::Mat equalisedintensity = Intensity;
		histogramEqualization(equalisedintensity, intensity, probabilityimg, cumulativeProbability);


		HSI_RGB(finalimg,equalisedintensity,Hue,Saturation);

		cv::namedWindow("Originalimage", cv::WINDOW_AUTOSIZE);
	    cv::imshow("Originalimage", img);
	    
	    cv::namedWindow("equalizedimage", cv::WINDOW_AUTOSIZE);
	    cv::imshow("equalizedimage", finalimg);

	    cv::imwrite( "/home/pritish/Downloads/equalizedimg3C.jpg", finalimg );


	    showHistogram(Intensity,"original");
	    showHistogram(equalisedintensity,"equalizedimg");
	}   
}

void histmatch(char* address,char* address2)
{
	cv::Mat img = cv::imread(address, CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat refimg = cv::imread(address2, CV_LOAD_IMAGE_UNCHANGED);
	

	int intensity[256] = { 0 };
    double probabilityimg[256] = { 0 };
    double cumulativeProbability[256] = { 0 };

    int intensityreq[256] = { 0 };
    double probabilityimgreq[256] = { 0 };
    double cumulativeProbabilityreq[256] = { 0 };

    if(img.channels() == 1)
    {	
	    
	    cv::Mat imgmatched = cv::Mat::zeros(img.rows,img.cols, CV_8UC1);
	    pixelFrequency(img, intensity);
	    pixelProbability(img, probabilityimg, intensity);
	    cumuProbability(probabilityimg, cumulativeProbability);

	    
	    pixelFrequency(refimg, intensityreq);
	    pixelProbability(refimg, probabilityimgreq, intensityreq);
	    cumuProbability(probabilityimgreq, cumulativeProbabilityreq);

	    int j = 1;
	    int finalIntensity[256] = {0};

	    if(cumulativeProbabilityreq == cumulativeProbability)
	    {
	    	std::cout<<"1";
	    }

	    for(int i = 0;i < 256;++i)
	    {
	    	while(cumulativeProbabilityreq[i]-cumulativeProbability[j] >= 0)
	    	{
	    		finalIntensity[j] = i;
	    		++j;
	    	}
	    }

	    
	    for(int i = 0;i < img.rows;++i)
	    {
	    	for(int j = 0;j < img.cols;++j)
	    	{
	    		imgmatched.at<uchar>(i, j) = finalIntensity[(int)img.at<uchar>(i, j)];
	    	}
	    }

	    

	    cv::namedWindow("targetimage", cv::WINDOW_AUTOSIZE);
	    cv::imshow("targetimage", refimg);

	    cv::namedWindow("matchedimage", cv::WINDOW_AUTOSIZE);
	    cv::imshow("matchedimage", imgmatched);

	    cv::imwrite( "/home/pritish/Downloads/matchedimg.jpg", imgmatched );


	    showHistogram(img,"original");
	    showHistogram(refimg,"required");
	    showHistogram(imgmatched,"matchedimg");

	}

	if(img.channels() >= 3)
	{
		cv::Mat imgmatched = cv::Mat::zeros(img.rows,img.cols, CV_8UC3);

		cv::Mat Intensity = cv::Mat::zeros(img.rows,img.cols, CV_8UC1);
		cv::Mat Hue = cv::Mat::zeros(img.rows,img.cols, CV_8UC1);
		cv::Mat Saturation = cv::Mat::zeros(cv::Size(img.rows,img.cols), CV_64FC1);

		RGB_HSI(img,Intensity,Hue,Saturation);

		cv::Mat Intensityreq = cv::Mat::zeros(refimg.rows,refimg.cols, CV_8UC1);
		cv::Mat Huereq = cv::Mat::zeros(refimg.rows,refimg.cols, CV_8UC1);
		cv::Mat Saturationreq = cv::Mat::zeros(cv::Size(refimg.rows,refimg.cols), CV_64FC1);

		RGB_HSI(refimg,Intensityreq,Huereq,Saturationreq);

		pixelFrequency(Intensity, intensity);
	    pixelProbability(Intensity, probabilityimg, intensity);
	    cumuProbability(probabilityimg, cumulativeProbability);

	    
	    pixelFrequency(Intensityreq, intensityreq);
	    pixelProbability(Intensityreq, probabilityimgreq, intensityreq);
	    cumuProbability(probabilityimgreq, cumulativeProbabilityreq);

	    cv::Mat intensitymatched = Intensity;

	    int j = 1;
	    int finalIntensity[256] = {0};

	    for(int i = 0;i < 256;++i)
	    {
	    	while(cumulativeProbabilityreq[i]-cumulativeProbability[j] >= 0)
	    	{
	    		finalIntensity[j] = i;
	    		++j;
	    	}
	    }

	    for(int i = 0;i < img.rows;++i)
	    {
	    	for(int j = 0;j < img.cols;++j)
	    	{
	    		intensitymatched.at<uchar>(i, j) = finalIntensity[int(intensitymatched.at<uchar>(i, j))];
	    	}
	    }

	    HSI_RGB(imgmatched,intensitymatched,Hue,Saturation);

	    cv::namedWindow("targetimage", cv::WINDOW_AUTOSIZE);
	    cv::imshow("targetimage", refimg);

	    cv::namedWindow("matchedimage", cv::WINDOW_AUTOSIZE);
	    cv::imshow("matchedimage", imgmatched);

	    cv::imwrite( "/home/pritish/Downloads/matchedimg3C.jpg", imgmatched );


	    showHistogram(Intensity,"original");
	    showHistogram(Intensityreq,"required");
	    showHistogram(intensitymatched,"matchedimg");
	}

}

int main(int argc,char** argv)
{
    histequalize(argv[1]);
    histmatch(argv[1],argv[2]);
    cv::waitKey(0);
    return 0;
}







