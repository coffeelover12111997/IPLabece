#include "fstream"
#include <iostream>
#include <string>
#include <vector>


void write_palette(FILE* fp){
	for(int i=0;i<256;i++){
		fwrite(&i,1,1,fp);
		fwrite(&i,1,1,fp);
		fwrite(&i,1,1,fp);
		fwrite(&i,1,1,fp);
	}
}

std::vector<unsigned char*> readBMP(char* filename)
{
	int i;
	FILE* f = fopen(filename, "rb");
	unsigned char* info = new unsigned char[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *((int*)&info[18]);
    int height = *((int*)&info[22]);

	int size = 3 * (width) * height;
	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	for (i = 0; i < size; i += 3)
	{
		unsigned char tmp = data[i];
		data[i] = data[i + 2];
		data[i + 2] = tmp;
	}

	std::vector<unsigned char*> v;
	v.push_back(info);
	v.push_back(data);
	return v;
}

void flipGrey(unsigned char* image,unsigned char* header)
{
	FILE* f = fopen("/home/pritish/Downloads/Exp-1/out.bmp", "wb");

	int width = *((int*)&header[18]);
	int height = *((int*)&header[22]);

	int size = 3 * width * height;
	int row_padded = (width*3 + 3) & (~3);

	unsigned char* finaldata = new unsigned char[size/3];

	int** outimage = new int*[height];
	for(int i = 0; i < height; ++i)
	    outimage[i] = new int[width];

	for (int i = 0; i < size; i += 3)
	{
		finaldata[i/3] = (image[i]+image[i + 1]+image[i + 2])/3;
	}

	
	*((int*)&header[2]) = 54+ 256*4 +(height*width);//sizeof bmp file

	*((int*)&header[10]) = 54+ 256*4 ;//offset of bmp data

	if(*((int*)&header[28]) != 8)//changing the bmp
	{
	 	*((int*)&header[28]) = 8;	
	}


	fwrite(header,sizeof(unsigned char),54,f);

	write_palette(f);

	// int k = 0;
	// for(int i = 0; i < height; ++i)
	// {
	// 	for(int j = 0;j < width; ++j)
	// 	{
	// 		outimage[i][j] = finaldata[k];
	// 		++k;
	// 	}
	// }

	// for(int i = 0; i < height; ++i)
	// {
	// 	for(int j = 0;j < i; ++j)
	// 	{
	// 		int temp = outimage[i][j];
	// 		outimage[i][j] = outimage[width-1-j][height-1-i];
	// 		outimage[width-1-j][height-1-i] = temp;
	// 	}
	// }

	// for(int t = 0;t < height;++t)
	// {
	// 	fwrite(outimage[t], sizeof(unsigned char), row_padded, f);	
	// }
	int x =  (width)*(height) - 1;
  	int y;
  	for(int i= 0;i< width;i++){
 	 y = x ;
  	for(int j= 0;j< height;j++){
    	fwrite(&(finaldata[y]),1,1,f);
		y = y - width;
  		}
    	x--;
  	}
}



int main(int argc,char** argv)
{

		const char* temp = "/home/pritish/Downloads/Exp-1/lena.bmp";
		std::vector<unsigned char*> bmpout = readBMP((char*)temp);
		flipGrey(bmpout[1],bmpout[0]);		
}