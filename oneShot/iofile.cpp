

#if defined(_MSC_VER)
#include <Windows.h>
#include <shlobj.h>
#include <Commdlg.h>
#include <ShellAPI.h>
#include <iostream>
#include "iofile.h"

namespace unre
{


	bool FileOP::FilesExist(CStr& fileW)
	{
		vecS names;
		int fNum = GetNames(fileW, names);
		return fNum > 0;
	}

	string FileOP::GetFolder(CStr& path)
	{
		return path.substr(0, path.find_last_of("\\/") + 1);
	}

	string FileOP::GetSubFolder(CStr& path)
	{
		string folder = path.substr(0, path.find_last_of("\\/"));
		return folder.substr(folder.find_last_of("\\/") + 1);
	}

	string FileOP::GetName(CStr& path)
	{
		int start = path.find_last_of("\\/") + 1;
		int end = path.find_last_not_of(' ') + 1;
		return path.substr(start, end - start);
	}

	string FileOP::GetNameNE(CStr& path)
	{
		int start = path.find_last_of("\\/") + 1;
		int end = path.find_last_of('.');
		if (end >= 0)
			return path.substr(start, end - start);
		else
			return path.substr(start, path.find_last_not_of(' ') + 1 - start);
	}

	string FileOP::GetNameNoSuffix(CStr& path, CStr &suffix)
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

	string FileOP::GetPathNE(CStr& path)
	{
		int end = path.find_last_of('.');
		if (end >= 0)
			return path.substr(0, end);
		else
			return path.substr(0, path.find_last_not_of(' ') + 1);
	}

	string FileOP::GetExtention(CStr name)
	{
		return name.substr(name.find_last_of('.'));
	}

	//bool FileOP::Copy(CStr &src, CStr &dst, bool failIfExist)
	//{
	//	return ::CopyFileA(src.c_str(), dst.c_str(), failIfExist);
	//}
	//
	//bool FileOP::Move(CStr &src, CStr &dst, DWORD dwFlags)
	//{
	//	return MoveFileExA(src.c_str(), dst.c_str(), dwFlags);
	//}
	//
	//void FileOP::RmFile(CStr& fileW)
	//{ 
	//	vecS names;
	//	string dir;
	//	int fNum = FileOP::GetNames(fileW, names, dir);
	//	for (int i = 0; i < fNum; i++)
	//		::DeleteFileA(_S(dir + names[i]));
	//}


	// Test whether a file exist




	string FileOP::GetWkDir()
	{
		string wd;
		wd.resize(1024);
		DWORD len = GetCurrentDirectoryA(1024, &wd[0]);
		wd.resize(len);
		return wd;
	}

	bool FileOP::FolderExist(CStr& strPath)
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


	bool FileOP::FileExist(CStr& filePath)
	{
		if (filePath.size() == 0)
			return false;
		DWORD attr = GetFileAttributesA(_S(filePath));
		return attr == FILE_ATTRIBUTE_NORMAL || attr == FILE_ATTRIBUTE_ARCHIVE;//GetLastError() != ERROR_FILE_NOT_FOUND;
	}
	bool FileOP::MkDir(CStr&  _path)
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


	string FileOP::BrowseFolder()
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

	string FileOP::BrowseFile(const char* strFilter, bool isOpen)
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

	//int FileOP::Rename(CStr& _srcNames, CStr& _dstDir, const char *nameCommon, const char *nameExt)
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
	//int FileOP::ChangeImgFormat(CStr &imgW, CStr dstW)
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
	//void FileOP::RenameSuffix(CStr dir, CStr orgSuf, CStr dstSuf)
	//{
	//	vecS namesNS;
	//	int fNum = FileOP::GetNamesNoSuffix(dir + "*" + orgSuf, namesNS, orgSuf);
	//	for (int i = 0; i < fNum; i++)
	//		FileOP::Move(dir + namesNS[i] + orgSuf, dir + namesNS[i] + dstSuf);
	//}
	//
	//void FileOP::RmFolder(CStr& dir)
	//{
	//	CleanFolder(dir);
	//	if (FolderExist(dir))
	//		RunProgram("Cmd.exe", cv::format("/c rmdir /s /q \"%s\"", _S(dir)), true, false);
	//}
	//
	//void FileOP::CleanFolder(CStr& dir, bool subFolder)
	//{
	//	vecS names;
	//	int fNum = FileOP::GetNames(dir + "/*.*", names);
	//	for (int i = 0; i < fNum; i++)
	//		RmFile(dir + "/" + names[i]);
	//
	//	vecS subFolders;
	//	int subNum = GetSubFolders(dir, subFolders);
	//	if (subFolder)
	//		for (int i = 0; i < subNum; i++)
	//			CleanFolder(dir + "/" + subFolders[i], true);
	//}

	int FileOP::GetSubFolders(CStr& folder, vecS& subFolders)
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
	int FileOP::GetNames(CStr &nameW, vecS &names, string &dir)
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
	//FileOP::GetNames(nameW, fileW, names);

	int FileOP::GetNames(CStr& rootFolder, CStr &fileW, vecS &names)
	{
		GetNames(rootFolder + fileW, names);
		vecS subFolders, tmpNames;
		int subNum = FileOP::GetSubFolders(rootFolder, subFolders);
		for (int i = 0; i < subNum; i++) {
			subFolders[i] += "/";
			int subNum = GetNames(rootFolder + subFolders[i], fileW, tmpNames);
			for (int j = 0; j < subNum; j++)
				names.push_back(subFolders[i] + tmpNames[j]);
		}
		return names.size();
	}




	int FileOP::GetFullName(CStr& folder, vecS& subFolders)
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
	FileOP::GetAbbreviateName(rootFolder, fileW, names);*/
	int FileOP::GetFullName(CStr & rootFolder, CStr & fileW, vecS & names)
	{
		GetNames(rootFolder + fileW, names);
		vecS subFolders, tmpNames;
		int subNum = FileOP::GetSubFolders(rootFolder, subFolders);
		for (int i = 0; i < subNum; i++) {
			subFolders[i] += "/";
			int subNum = GetFullName(rootFolder + subFolders[i], fileW, tmpNames);
			for (int j = 0; j < subNum; j++)
				names.push_back(tmpNames[j]);
		}
		return names.size();
	}

	int FileOP::GetFullPath(CStr& rootFolder, CStr &fileW, vecS &names)
	{
		GetNames(rootFolder, fileW, names);
		for (auto& name : names)
		{
			name = rootFolder + name;
		}
		return names.size();
	}
	int FileOP::GetNameNoSuffix(CStr & rootFolder, CStr & fileW, vecS & names)
	{
		int fNum = GetFullName(rootFolder, fileW, names);
		int extS = (int)GetExtention(fileW).size();
		for (int i = 0; i < fNum; i++)
			names[i].resize(names[i].size() - extS);
		return fNum;
	}

}
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

//#  include"../log/logging.h"

namespace unre
{
	bool FileOP::FilesExist(CStr& fileW)
	{
		return access(fileW.c_str(), F_OK) == 0;
	}

	string FileOP::GetFolder(CStr& path)
	{
		return path.substr(0, path.find_last_of("\\/") + 1);
	}

	string FileOP::GetSubFolder(CStr& path)
	{
		string folder = path.substr(0, path.find_last_of("\\/"));
		return folder.substr(folder.find_last_of("\\/") + 1);
	}

	string FileOP::GetName(CStr& path)
	{
		int start = path.find_last_of("\\/") + 1;
		int end = path.find_last_not_of(' ') + 1;
		return path.substr(start, end - start);
	}

	string FileOP::GetNameNE(CStr& path)
	{
		int start = path.find_last_of("\\/") + 1;
		int end = path.find_last_of('.');
		if (end >= 0)
			return path.substr(start, end - start);
		else
			return path.substr(start, path.find_last_not_of(' ') + 1 - start);
	}

	string FileOP::GetNameNoSuffix(CStr& path, CStr &suffix)
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

	string FileOP::GetPathNE(CStr& path)
	{
		int end = path.find_last_of('.');
		if (end >= 0)
			return path.substr(0, end);
		else
			return path.substr(0, path.find_last_not_of(' ') + 1);
	}

	string FileOP::GetExtention(CStr name)
	{
		return name.substr(name.find_last_of('.'));
	}


	string FileOP::GetWkDir()
	{
		char buf[200];
		return(std::string(getcwd(buf, 200)));
	}

	bool FileOP::FolderExist(CStr& strPath)
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


	bool FileOP::FileExist(CStr& filePath)
	{
		return !access(filePath.c_str(), F_OK);
	}
	bool FileOP::MkDir(CStr&  _path)
	{
		if (_path.size() == 0)
			return false;
		int ret = mkdir(_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH | S_IWOTH);
		return (ret == 0);
	}


	string FileOP::BrowseFolder()
	{
		NOT_IMPLEMENTED;
		return "";
	}

	string FileOP::BrowseFile(const char* strFilter, bool isOpen)
	{
		NOT_IMPLEMENTED;
		return "";
	}


	int FileOP::GetSubFolders(CStr& folder, vecS& subFolders)
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

	int FileOP::GetNames(CStr &nameW, vecS &names, string &dir)
	{
		NOT_IMPLEMENTED;
		return 0;
	}

	int FileOP::GetNames(CStr& rootFolder, CStr &fileW, vecS &names)
	{
		NOT_IMPLEMENTED;
		return 0;
	}


	int FileOP::GetFullName(CStr& folder, vecS& subFolders)
	{
		NOT_IMPLEMENTED;
		return 0;
	}

	int FileOP::GetFullName(CStr & rootFolder, CStr & fileW, vecS & names)
	{
		NOT_IMPLEMENTED;
		return 0;
	}

	int FileOP::GetFullPath(CStr& rootFolder_, CStr &fileW, vecS &names)
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
				FileOP::GetFullPath(thisFile, fileW, names);
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
	int FileOP::GetNameNoSuffix(CStr & rootFolder, CStr & fileW, vecS & names)
	{
		NOT_IMPLEMENTED;
		return 0;
	}

}

#else
#  error "Platform not supported!"
#endif





