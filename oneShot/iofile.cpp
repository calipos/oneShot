#pragma once  

#if defined(_MSC_VER)
#include <Windows.h>
#include <shlobj.h>
#include <Commdlg.h>
#include <ShellAPI.h>
#include <iostream>
#include "iofile.h"


//#include "opencv2/core/utility.hpp"
//#include "opencv/opencv.hpp"

/************************************************************************/
/* Implementation of inline functions                                   */
/************************************************************************/

bool UNREFile::FilesExist(CStr& fileW)
{
	vecS names;
	int fNum = GetNames(fileW, names);
	return fNum > 0;
}

string UNREFile::GetFolder(CStr& path)
{
	return path.substr(0, path.find_last_of("\\/") + 1);
}

string UNREFile::GetSubFolder(CStr& path)
{
	string folder = path.substr(0, path.find_last_of("\\/"));
	return folder.substr(folder.find_last_of("\\/") + 1);
}

string UNREFile::GetName(CStr& path)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.find_last_not_of(' ') + 1;
	return path.substr(start, end - start);
}

string UNREFile::GetNameNE(CStr& path)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(start, end - start);
	else
		return path.substr(start, path.find_last_not_of(' ') + 1 - start);
}

string UNREFile::GetNameNoSuffix(CStr& path, CStr &suffix)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.size() - suffix.size();
	//CHECK(path.substr(end) == suffix);
	//CV_Assert(path.substr(end) == suffix);
	if (end >= 0)
		return path.substr(start, end - start);
	else
		return path.substr(start, path.find_last_not_of(' ') + 1 - start);
}

string UNREFile::GetPathNE(CStr& path)
{
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(0, end);
	else
		return path.substr(0, path.find_last_not_of(' ') + 1);
}

string UNREFile::GetExtention(CStr name)
{
	return name.substr(name.find_last_of('.'));
}

//bool UNREFile::Copy(CStr &src, CStr &dst, bool failIfExist)
//{
//	return ::CopyFileA(src.c_str(), dst.c_str(), failIfExist);
//}
//
//bool UNREFile::Move(CStr &src, CStr &dst, DWORD dwFlags)
//{
//	return MoveFileExA(src.c_str(), dst.c_str(), dwFlags);
//}
//
//void UNREFile::RmFile(CStr& fileW)
//{ 
//	vecS names;
//	string dir;
//	int fNum = UNREFile::GetNames(fileW, names, dir);
//	for (int i = 0; i < fNum; i++)
//		::DeleteFileA(_S(dir + names[i]));
//}


// Test whether a file exist




string UNREFile::GetWkDir()
{
	string wd;
	wd.resize(1024);
	DWORD len = GetCurrentDirectoryA(1024, &wd[0]);
	wd.resize(len);
	return wd;
}

bool UNREFile::FolderExist(CStr& strPath)
{
	int i = (int)strPath.size() - 1;
	for (; i >= 0 && (strPath[i] == '\\' || strPath[i] == '/'); i--)
		;
	string str = strPath.substr(0, i + 1);

	WIN32_FIND_DATAA  wfd;
	HANDLE hFind = FindFirstFileA(_S(str), &wfd);
	bool rValue = (hFind != INVALID_HANDLE_VALUE) && (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
	FindClose(hFind);
	return rValue;
}

/************************************************************************/
/*                   Implementations                                    */
/************************************************************************/


bool UNREFile::FileExist(CStr& filePath)
{
	if (filePath.size() == 0)
		return false;
	DWORD attr = GetFileAttributesA(_S(filePath));
	return attr == FILE_ATTRIBUTE_NORMAL || attr == FILE_ATTRIBUTE_ARCHIVE;//GetLastError() != ERROR_FILE_NOT_FOUND;
}
bool UNREFile::MkDir(CStr&  _path)
{
	if (_path.size() == 0)
		return false;

	static char buffer[1024];
	strcpy(buffer, _S(_path));
	for (int i = 0; buffer[i] != 0; i++) {
		if (buffer[i] == '\\' || buffer[i] == '/') {
			buffer[i] = '\0';
			CreateDirectoryA(buffer, 0);
			buffer[i] = '/';
		}
	}
	return CreateDirectoryA(_S(_path), 0);
}


string UNREFile::BrowseFolder()
{
	static char Buffer[MAX_PATH];
	BROWSEINFOA bi;//Initial bi 	
	bi.hwndOwner = NULL;
	bi.pidlRoot = NULL;
	bi.pszDisplayName = Buffer; // Dialog can't be shown if it's NULL
	bi.lpszTitle = "BrowseFolder";
	bi.ulFlags = 0;
	bi.lpfn = NULL;
	bi.iImage = NULL;


	LPITEMIDLIST pIDList = SHBrowseForFolderA(&bi); // Show dialog
	if (pIDList) {
		SHGetPathFromIDListA(pIDList, Buffer);
		if (Buffer[strlen(Buffer) - 1] == '\\')
			Buffer[strlen(Buffer) - 1] = 0;

		return string(Buffer);
	}
	return string();
}

string UNREFile::BrowseFile(const char* strFilter, bool isOpen)
{
	static char Buffer[MAX_PATH];
	OPENFILENAMEA   ofn;
	memset(&ofn, 0, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.lpstrFile = Buffer;
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFilter = strFilter;
	ofn.nFilterIndex = 1;
	ofn.Flags = OFN_PATHMUSTEXIST;

	if (isOpen) {
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
		GetOpenFileNameA(&ofn);
		return Buffer;
	}

	GetSaveFileNameA(&ofn);
	return string(Buffer);

}

//int UNREFile::Rename(CStr& _srcNames, CStr& _dstDir, const char *nameCommon, const char *nameExt)
//{
//	vecS names;
//	string inDir;
//	int fNum = GetNames(_srcNames, names, inDir);
//	for (int i = 0; i < fNum; i++) {
//		string dstName = cv::format("%s\\%.4d%s.%s", _S(_dstDir), i, nameCommon, nameExt);
//		string srcName = inDir + names[i];
//		::CopyFileA(srcName.c_str(), dstName.c_str(), FALSE);
//	}
//	return fNum;
//}
//
//int UNREFile::ChangeImgFormat(CStr &imgW, CStr dstW)
//{
//	vecS names;
//	string inDir, ext = GetExtention(imgW);
//	int iNum = GetNames(imgW, names, inDir);
//#pragma omp parallel for
//	for (int i = 0; i < iNum; i++) {
//		Mat img = imread(inDir + names[i]);
//		imwrite(cv::format(_S(dstW), _S(GetNameNE(names[i]))), img);
//	}
//	return iNum;
//}
//
//void UNREFile::RenameSuffix(CStr dir, CStr orgSuf, CStr dstSuf)
//{
//	vecS namesNS;
//	int fNum = UNREFile::GetNamesNoSuffix(dir + "*" + orgSuf, namesNS, orgSuf);
//	for (int i = 0; i < fNum; i++)
//		UNREFile::Move(dir + namesNS[i] + orgSuf, dir + namesNS[i] + dstSuf);
//}
//
//void UNREFile::RmFolder(CStr& dir)
//{
//	CleanFolder(dir);
//	if (FolderExist(dir))
//		RunProgram("Cmd.exe", cv::format("/c rmdir /s /q \"%s\"", _S(dir)), true, false);
//}
//
//void UNREFile::CleanFolder(CStr& dir, bool subFolder)
//{
//	vecS names;
//	int fNum = UNREFile::GetNames(dir + "/*.*", names);
//	for (int i = 0; i < fNum; i++)
//		RmFile(dir + "/" + names[i]);
//
//	vecS subFolders;
//	int subNum = GetSubFolders(dir, subFolders);
//	if (subFolder)
//		for (int i = 0; i < subNum; i++)
//			CleanFolder(dir + "/" + subFolders[i], true);
//}

int UNREFile::GetSubFolders(CStr& folder, vecS& subFolders)
{
	subFolders.clear();
	WIN32_FIND_DATAA fileFindData;
	string nameWC = folder + "\\*";
	HANDLE hFind = ::FindFirstFileA(nameWC.c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	do {
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			subFolders.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)subFolders.size();
}

// Get image names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
int UNREFile::GetNames(CStr &nameW, vecS &names, string &dir)
{
	dir = GetFolder(nameW);
	names.clear();
	names.reserve(10000);
	WIN32_FIND_DATAA fileFindData;
	HANDLE hFind = ::FindFirstFileA(_S(nameW), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	do {
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue; // Ignore sub-folders
		names.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)names.size();
}
//
//CStr nameW = "G:\\Data\\";
//vecS names;
//CStr fileW = "*.jpg";
//UNREFile::GetNames(nameW, fileW, names);

int UNREFile::GetNames(CStr& rootFolder, CStr &fileW, vecS &names)
{
	GetNames(rootFolder + fileW, names);
	vecS subFolders, tmpNames;
	int subNum = UNREFile::GetSubFolders(rootFolder, subFolders);
	for (int i = 0; i < subNum; i++) {
		subFolders[i] += "/";
		int subNum = GetNames(rootFolder + subFolders[i], fileW, tmpNames);
		for (int j = 0; j < subNum; j++)
			names.push_back(subFolders[i] + tmpNames[j]);
	}
	return names.size();
}




int UNREFile::GetFullName(CStr& folder, vecS& subFolders)
{
	subFolders.clear();
	WIN32_FIND_DATAA fileFindData;
	string nameWC = folder + "\\*";
	HANDLE hFind = ::FindFirstFileA(nameWC.c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	do {
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			subFolders.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)subFolders.size();
}

/*
CStr rootFolder = "G:/Data/";
vecS names;
CStr fileW = "*.jpg";
UNREFile::GetAbbreviateName(rootFolder, fileW, names);*/
int UNREFile::GetFullName(CStr & rootFolder, CStr & fileW, vecS & names)
{
	GetNames(rootFolder + fileW, names);
	vecS subFolders, tmpNames;
	int subNum = UNREFile::GetSubFolders(rootFolder, subFolders);
	for (int i = 0; i < subNum; i++) {
		subFolders[i] += "/";
		int subNum = GetFullName(rootFolder + subFolders[i], fileW, tmpNames);
		for (int j = 0; j < subNum; j++)
			names.push_back(tmpNames[j]);
	}
	return names.size();
}

int UNREFile::GetFullPath(CStr& rootFolder, CStr &fileW, vecS &names)
{
	GetNames(rootFolder, fileW, names);
	for (auto& name : names)
	{
		name = rootFolder + name;
	}
	return names.size();
}
int UNREFile::GetNameNoSuffix(CStr & rootFolder, CStr & fileW, vecS & names)
{
	int fNum = GetFullName(rootFolder, fileW, names);
	int extS = (int)GetExtention(fileW).size();
	for (int i = 0; i < fNum; i++)
		names[i].resize(names[i].size() - extS);
	return fNum;
}



//
//
//CStr rootFolder = "G:/Data/";
//vecS names;
//CStr fileW = "*.jpg";
//UNREFile::GetNamesNE(rootFolder, fileW, names);
//int UNREFile::GetNamesNE(CStr& nameWC, vecS &names, string &dir, string &ext)
//{
//	int fNum = GetNames(nameWC, names, dir);
//	ext = GetExtention(nameWC);
//	for (int i = 0; i < fNum; i++)
//		names[i] = GetNameNE(names[i]);
//	return fNum;
//}
//
//int UNREFile::GetNamesNE(CStr& rootFolder, CStr &fileW, vecS &names)
//{
//	int fNum = GetNames(rootFolder, fileW, names);
//	int extS = (int)GetExtention(fileW).size();
//	for (int i = 0; i < fNum; i++)
//		names[i].resize(names[i].size() - extS);
//	return fNum;
//}
//
//int UNREFile::GetNamesNoSuffix(CStr& nameWC, vecS &namesNS, CStr suffix, string &dir)
//{
//	int fNum = UNREFile::GetNames(nameWC, namesNS, dir);
//	for (int i = 0; i < fNum; i++)
//		namesNS[i] = GetNameNoSuffix(namesNS[i], suffix);
//	return fNum;
//}

//// Load mask image and threshold thus noisy by compression can be removed
//Mat UNREFile::LoadMask(CStr& fileName)
//{
//	Mat mask = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
//	CV_Assert_(mask.data != NULL, ("Can't find mask image: %s", _S(fileName)));
//	compare(mask, 128, mask, CV_CMP_GT);
//	return mask;
//}

//BOOL UNREFile::Move2Dir(CStr &srcW, CStr dstDir)
//{
//	vecS names;
//	string inDir;
//	int fNum = UNREFile::GetNames(srcW, names, inDir);
//	BOOL r = TRUE;
//	for (int i = 0; i < fNum; i++)	
//		if (Move(inDir + names[i], dstDir + names[i]) == FALSE)
//			r = FALSE;
//	return r;
//}
//
//BOOL UNREFile::Copy2Dir(CStr &srcW, CStr dstDir)
//{
//	vecS names;
//	string inDir;
//	int fNum = UNREFile::GetNames(srcW, names, inDir);
//	BOOL r = TRUE;
//	for (int i = 0; i < fNum; i++)	
//		if (Copy(inDir + names[i], dstDir + names[i]) == FALSE)
//			r = FALSE;
//	return r;
//}
//
//void UNREFile::ChkImgs(CStr &imgW)
//{
//	vecS names;
//	string inDir;
//	int imgNum = GetNames(imgW, names, inDir);
//	printf("Checking %d images: %s\n", imgNum, _S(imgW));
//	for (int i = 0; i < imgNum; i++){
//		Mat img = imread(inDir + names[i]);
//		if (img.data == NULL)
//			printf("Loading file %s failed\t\t\n", _S(names[i]));
//		if (i % 200 == 0)
//			printf("Processing %2.1f%%\r", (i*100.0)/imgNum);
//	}
//	printf("\t\t\t\t\r");
//}
//
//
//void UNREFile::AppendStr(CStr fileName, CStr str)
//{
//	FILE *f = fopen(_S(fileName), "a");
//	if (f == NULL){
//		printf("File %s can't be opened\n", _S(fileName));
//		return;
//	}
//	fprintf(f, "%s", _S(str));
//	fclose(f);
//}
//
//
//void UNREFile::RunProgram(CStr &fileName, CStr &parameters, bool waiteF, bool showW)
//{
//	string runExeFile = fileName;
//#ifdef _DEBUG
//	runExeFile.insert(0, "..\\Debug\\");
//#else
//	runExeFile.insert(0, "..\\Release\\");
//#endif // _DEBUG
//	if (!UNREFile::FileExist(_S(runExeFile)))
//		runExeFile = fileName;
//
//	SHELLEXECUTEINFOA  ShExecInfo  =  {0};  
//	ShExecInfo.cbSize  =  sizeof(SHELLEXECUTEINFO);  
//	ShExecInfo.fMask  =  SEE_MASK_NOCLOSEPROCESS;  
//	ShExecInfo.hwnd  =  NULL;  
//	ShExecInfo.lpVerb  =  NULL;  
//	ShExecInfo.lpFile  =  _S(runExeFile);
//	ShExecInfo.lpParameters  =  _S(parameters);         
//	ShExecInfo.lpDirectory  =  NULL;  
//	ShExecInfo.nShow  =  showW ? SW_SHOW : SW_HIDE;  
//	ShExecInfo.hInstApp  =  NULL;              
//	ShellExecuteExA(&ShExecInfo);  
//
//	//printf("Run: %s %s\n", ShExecInfo.lpFile, ShExecInfo.lpParameters);
//
//	if (waiteF)
//		WaitForSingleObject(ShExecInfo.hProcess,INFINITE);
//}
//
//string UNREFile::GetCompName() 
//{
//	char buf[1024];
//	DWORD dwCompNameLen = 1024;
//	GetComputerNameA(buf, &dwCompNameLen);
//	return string(buf);
//}
//
//void UNREFile::SegOmpThrdNum(double ratio /* = 0.8 */)
//{
//	int thrNum = omp_get_max_threads();
//	int usedNum = cvRound(thrNum * ratio);
//	usedNum = std::max(usedNum, 1);
//	//printf("Number of CPU cores used is %d/%d\n", usedNum, thrNum);
//	omp_set_num_threads(usedNum);
//}
//
//
//// Copy files and add suffix. e.g. copyAddSuffix("./*.jpg", "./Imgs/", "_Img.jpg")
//void UNREFile::copyAddSuffix(CStr &srcW, CStr &dstDir, CStr &dstSuffix)
//{
//	vecS namesNE;
//	string srcDir, srcExt;
//	int imgN = UNREFile::GetNamesNE(srcW, namesNE, srcDir, srcExt);
//	UNREFile::MkDir(dstDir);
//	for (int i = 0; i < imgN; i++)
//		UNREFile::Copy(srcDir + namesNE[i] + srcExt, dstDir + namesNE[i] + dstSuffix);
//}
//
//vecS UNREFile::loadStrList(CStr &fName)
//{
//	std::ifstream fIn(fName);
//	string line;
//	vecS strs;
//	while(getline(fIn, line) && line.size())
//		strs.push_back(line);
//	return strs;
//}


//// Write matrix to binary file
//bool UNREFile::matWrite(CStr& filename, CMat& M){
//	FILE* f = fopen(_S(filename), "wb");
//	bool res = matWrite(f, M);
//	if (f != NULL)
//		fclose(f);	
//	return res;
//}
//
//bool UNREFile::matWrite(FILE *f, CMat& _M)
//{
//	Mat M;
//	_M.copyTo(M);
//	if (f == NULL || M.empty())
//		return false;
//	fwrite("CmMat", sizeof(char), 5, f);
//	int headData[3] = {M.cols, M.rows, M.type()};
//	fwrite(headData, sizeof(int), 3, f);
//	fwrite(M.data, sizeof(char), M.step * M.rows, f);
//	return true;
//}

///****************************************************************************/
//// Read matrix from binary file
//bool UNREFile::matRead( const string& filename, Mat& M){
//	FILE* f = fopen(_S(filename), "rb");
//	bool res = matRead(f, M);
//	if (f != NULL)
//		fclose(f);
//	return res;
//}
//
//bool UNREFile::matRead(FILE *f, Mat& M)
//{
//	if (f == NULL)
//		return false;
//	char buf[8];
//	int pre = (int)fread(buf,sizeof(char), 5, f);
//	if (strncmp(buf, "CmMat", 5) != 0)	{
//		printf("Invalidate CvMat data file: %d:%s\n", __LINE__, __FILE__);
//		return false;
//	}
//	int headData[3]; // Width, height, type
//	fread(headData, sizeof(int), 3, f);
//	M = Mat(headData[1], headData[0], headData[2]);
//	fread(M.data, sizeof(char), M.step * M.rows, f);
//	return true;
//}
//
//void UNREFile::ZipFiles(CStr &filesW, CStr &zipFileName, int compressLevel)
//{
//	string param = std::format("u -tzip -mmt -mx%d \"%s\" \"%s\"", compressLevel, _S(zipFileName), _S(filesW));
//	printf("Zip files: %s --> %s\n", _S(filesW), _S(zipFileName));
//	RunProgram("7z.exe", param, true, false);
//}
//
//
//void UNREFile::UnZipFiles(CStr &zipFileName, CStr &tgtDir, bool overwriteWarning/* = true*/)
//{
//	string param = std::format("e \"%s\" \"-o%s\" -r", _S(zipFileName), _S(tgtDir));
//	if (!overwriteWarning)
//		param += " -y";
//	if (!FileExist(zipFileName))
//		printf("File missing: %s\n", _S(zipFileName));
//
//	if (overwriteWarning)
//		printf("UnZip files: %s --> %s\n", _S(zipFileName), _S(tgtDir));
//	UNREFile::RunProgram("7z.exe", param, true, overwriteWarning);
//}
#elif defined __GNUC__ && __GNUC__ >= 4
#  include <stdlib.h>  
#  include <stdio.h>  
#  include <unistd.h>  
#  include <sys/types.h>  
#  include <sys/stat.h>  
#  include <dirent.h>
#  include <fcntl.h>  
#  include <string.h>  
#  include <errno.h>  
#  include <fnmatch.h>  
#  include<iostream>
#  include "iofile.h"

#  include"../log/logging.h"
//#include "opencv2/core/utility.hpp"
//#include "opencv/opencv.hpp"

/************************************************************************/
/* Implementation of inline functions                                   */
/************************************************************************/

bool UNREFile::FilesExist(CStr& fileW)
{
	return access(fileW.c_str(), F_OK) == 0;
}

string UNREFile::GetFolder(CStr& path)
{
	return path.substr(0, path.find_last_of("\\/") + 1);
}

string UNREFile::GetSubFolder(CStr& path)
{
	string folder = path.substr(0, path.find_last_of("\\/"));
	return folder.substr(folder.find_last_of("\\/") + 1);
}

string UNREFile::GetName(CStr& path)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.find_last_not_of(' ') + 1;
	return path.substr(start, end - start);
}

string UNREFile::GetNameNE(CStr& path)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(start, end - start);
	else
		return path.substr(start, path.find_last_not_of(' ') + 1 - start);
}

string UNREFile::GetNameNoSuffix(CStr& path, CStr &suffix)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.size() - suffix.size();
	//CHECK(path.substr(end) == suffix);
	//CV_Assert(path.substr(end) == suffix);
	if (end >= 0)
		return path.substr(start, end - start);
	else
		return path.substr(start, path.find_last_not_of(' ') + 1 - start);
}

string UNREFile::GetPathNE(CStr& path)
{
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(0, end);
	else
		return path.substr(0, path.find_last_not_of(' ') + 1);
}

string UNREFile::GetExtention(CStr name)
{
	return name.substr(name.find_last_of('.'));
}


string UNREFile::GetWkDir()
{
	char buf[200];
	return(std::string(getcwd(buf, 200)));
}

bool UNREFile::FolderExist(CStr& strPath)
{
	if (strPath.empty()) return false;
	DIR *pDir;
	bool bExists = false;
	pDir = opendir(strPath.c_str());
	if (pDir != NULL)
	{
		bExists = true;
		(void)closedir(pDir);
	}
	return bExists;
}

/************************************************************************/
/*                   Implementations                                    */
/************************************************************************/


bool UNREFile::FileExist(CStr& filePath)
{
	return !access(filePath.c_str(), F_OK);
}
bool UNREFile::MkDir(CStr&  _path)
{
	if (_path.size() == 0)
		return false;
	int ret = mkdir(_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH | S_IWOTH);
	return (ret == 0);
}


string UNREFile::BrowseFolder()
{
	NOT_IMPLEMENTED;
	return "";
}

string UNREFile::BrowseFile(const char* strFilter, bool isOpen)
{
	NOT_IMPLEMENTED;
	return "";
}


int UNREFile::GetSubFolders(CStr& folder, vecS& subFolders)
{
	// check the parameter !  
	if (NULL == folder.c_str())
	{
		LOG(WARNING) << " dir_name is null ! ";
		return 0;
	}
	struct stat s;
	lstat(folder.c_str(), &s);
	if (!S_ISDIR(s.st_mode))
	{
		LOG(WARNING) << "dir_name is not a valid directory !";
		return 0;
	}
	struct dirent * filename;    // return value for readdir()  
	DIR * dir;                   // return value for opendir()  
	dir = opendir(folder.c_str());
	if (NULL == dir)
	{
		LOG(WARNING) << "Can not open dir " << folder;
		return 0;
	}
	while ((filename = readdir(dir)) != NULL)
	{
		// get rid of "." and ".."  
		if (strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0)
			continue;
		lstat(filename->d_name, &s);
		if (S_ISDIR(s.st_mode))
		{
			subFolders.push_back(std::string(filename->d_name));
		}
	}
	return (int)subFolders.size();
}

int UNREFile::GetNames(CStr &nameW, vecS &names, string &dir)
{
	NOT_IMPLEMENTED;
	return 0;
}

int UNREFile::GetNames(CStr& rootFolder, CStr &fileW, vecS &names)
{
	NOT_IMPLEMENTED;
	return 0;
}


int UNREFile::GetFullName(CStr& folder, vecS& subFolders)
{
	NOT_IMPLEMENTED;
	return 0;
}

int UNREFile::GetFullName(CStr & rootFolder, CStr & fileW, vecS & names)
{
	NOT_IMPLEMENTED;
	return 0;
}

int UNREFile::GetFullPath(CStr& rootFolder_, CStr &fileW, vecS &names)
{
	std::string rootFolder;
	if (rootFolder_[rootFolder_.size() - 1] == '/')
	{
		rootFolder = rootFolder_.substr(0, rootFolder_.size() - 1);
	}
	else
	{
		rootFolder = rootFolder_;
	}
	if (NULL == rootFolder.c_str())
	{
		LOG(WARNING) << " dir_name is null ! ";
		return -1;
	}
	struct stat s;
	lstat(rootFolder.c_str(), &s);
	if (!S_ISDIR(s.st_mode))
	{
		LOG(WARNING) << rootFolder << " is not a valid directory !";
		return -1;
	}
	struct dirent * filename;    // return value for readdir()  
	DIR * dir;                   // return value for opendir()  
	dir = opendir(rootFolder.c_str());
	if (NULL == dir)
	{
		LOG(WARNING) << "Can not open dir " << rootFolder;
		return 0;
	}
	while ((filename = readdir(dir)) != NULL)
	{
		if (strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0)
			continue;
		std::string thisFile = rootFolder + "/" + filename->d_name;

		lstat(thisFile.c_str(), &s);
		if (S_ISDIR(s.st_mode))
		{
			UNREFile::GetFullPath(thisFile, fileW, names);
		}
		else if (0 == fnmatch(fileW.c_str(), filename->d_name, FNM_PATHNAME | FNM_PERIOD))
		{
			names.emplace_back(std::move(thisFile));
		}
		else
		{
			continue;
		}
	}
	return 0;
}
int UNREFile::GetNameNoSuffix(CStr & rootFolder, CStr & fileW, vecS & names)
{
	NOT_IMPLEMENTED;
	return 0;
}



#else
#  error "Platform not supported!"
#endif





