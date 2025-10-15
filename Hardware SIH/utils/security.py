import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import hashlib
import hmac
import secrets
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    # Create dummy classes for type checking
    class Fernet:
        def __init__(self, *args, **kwargs):
            pass
        @classmethod
        def generate_key(cls):
            return b'dummy_key_for_testing'
        def encrypt(self, data):
            return b'dummy_encrypted_data'
        def decrypt(self, data):
            return b'dummy_decrypted_data'
    class hashes:
        class SHA256:
            pass
    class PBKDF2HMAC:
        def __init__(self, *args, **kwargs):
            pass
    CRYPTO_AVAILABLE = False
import sqlite3

class SecurityManager:
    """
    A class to manage security features for the R&D proposal evaluation system
    """
    
    def __init__(self, db_path: str = "security.db"):
        """
        Initialize the security manager
        
        Args:
            db_path (str): Path to the security database
        """
        self.db_path = db_path
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.init_security_database()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """
        Get or create encryption key
        
        Returns:
            bytes: Encryption key
        """
        key_file = "encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            return key
    
    def init_security_database(self) -> None:
        """
        Initialize the security database schema
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Create roles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS roles (
                role_id TEXT PRIMARY KEY,
                role_name TEXT UNIQUE NOT NULL,
                permissions TEXT,  -- JSON string of permissions
                description TEXT
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create audit_log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                action TEXT,
                resource TEXT,
                details TEXT,  -- JSON string with additional details
                ip_address TEXT,
                success BOOLEAN
            )
        ''')
        
        # Create access_control table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_control (
                ac_id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_type TEXT,
                resource_id TEXT,
                user_id TEXT,
                permission_level TEXT,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                granted_by TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (granted_by) REFERENCES users (user_id)
            )
        ''')
        
        # Insert default roles if they don't exist
        default_roles = [
            ('admin', 'Administrator', ['all'], 'Full system access'),
            ('reviewer', 'Proposal Reviewer', ['read_proposals', 'evaluate', 'feedback'], 'Review proposal access'),
            ('submitter', 'Proposal Submitter', ['submit_proposals', 'view_own_proposals'], 'Submit proposal access'),
            ('viewer', 'Read-only Viewer', ['read_proposals'], 'View proposals access')
        ]
        
        for role_id, role_name, permissions, description in default_roles:
            cursor.execute('''
                INSERT OR IGNORE INTO roles (role_id, role_name, permissions, description)
                VALUES (?, ?, ?, ?)
            ''', (role_id, role_name, json.dumps(permissions), description))
        
        # Insert default admin user if it doesn't exist
        admin_password = "Admin123!"  # In production, this should be set securely
        if not self.user_exists("admin"):
            self.create_user("admin", "admin", admin_password, "admin", "admin@system.org")
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Hash a password with salt
        
        Args:
            password (str): Password to hash
            salt (Optional[bytes]): Salt to use (generate new if None)
            
        Returns:
            Tuple[str, str]: (hashed_password, salt_as_string)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        salt_b64 = base64.b64encode(salt).decode('utf-8')
        password_hash = hmac.new(salt, password.encode('utf-8'), hashlib.sha256).hexdigest()
        
        return password_hash, salt_b64
    
    def verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
        """
        Verify a password against stored hash and salt
        
        Args:
            password (str): Password to verify
            stored_hash (str): Stored password hash
            stored_salt (str): Stored salt
            
        Returns:
            bool: True if password is correct
        """
        salt = base64.b64decode(stored_salt.encode('utf-8'))
        password_hash = hmac.new(salt, password.encode('utf-8'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(password_hash, stored_hash)
    
    def create_user(self, user_id: str, username: str, password: str, 
                   role: str, email: Optional[str] = None) -> bool:
        """
        Create a new user
        
        Args:
            user_id (str): User ID
            username (str): Username
            password (str): Password
            role (str): User role
            email (Optional[str]): User email
            
        Returns:
            bool: True if user created successfully
        """
        # Check if user already exists
        if self.user_exists(username):
            return False
        
        # Hash password
        password_hash, salt = self.hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (user_id, username, password_hash, salt, role, email)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, password_hash, salt, role, email))
            
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def user_exists(self, username: str) -> bool:
        """
        Check if a user exists
        
        Args:
            username (str): Username to check
            
        Returns:
            bool: True if user exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT 1 FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, str]]:
        """
        Authenticate a user
        
        Args:
            username (str): Username
            password (str): Password
            
        Returns:
            Optional[Dict[str, str]]: User info if authentication successful, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, password_hash, salt, role, email, is_active
            FROM users WHERE username = ?
        ''', (username,))
        
        user_row = cursor.fetchone()
        conn.close()
        
        if not user_row:
            return None
        
        user_id, username, stored_hash, stored_salt, role, email, is_active = user_row
        
        if not is_active:
            return None
        
        if self.verify_password(password, stored_hash, stored_salt):
            # Update last login
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
            conn.close()
            
            return {
                'user_id': user_id,
                'username': username,
                'role': role,
                'email': email
            }
        
        return None
    
    def create_session(self, user_id: str, ip_address: str = "", 
                      user_agent: str = "") -> str:
        """
        Create a new session for a user
        
        Args:
            user_id (str): User ID
            ip_address (str): IP address of the user
            user_agent (str): User agent string
            
        Returns:
            str: Session ID
        """
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)  # 24-hour session
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, user_id, expires_at, ip_address, user_agent))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, str]]:
        """
        Validate a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            Optional[Dict[str, str]]: User info if session is valid, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.user_id, u.username, u.role, u.email, s.expires_at
            FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.session_id = ? AND s.is_active = TRUE AND s.expires_at > CURRENT_TIMESTAMP
        ''', (session_id,))
        
        session_row = cursor.fetchone()
        conn.close()
        
        if not session_row:
            return None
        
        user_id, username, role, email, expires_at = session_row
        
        return {
            'user_id': user_id,
            'username': username,
            'role': role,
            'email': email
        }
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            bool: True if session was invalidated
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE sessions SET is_active = FALSE WHERE session_id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if a user has a specific permission
        
        Args:
            user_id (str): User ID
            permission (str): Permission to check
            
        Returns:
            bool: True if user has permission
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user role
        cursor.execute('SELECT role FROM users WHERE user_id = ?', (user_id,))
        role_row = cursor.fetchone()
        
        if not role_row:
            conn.close()
            return False
        
        role = role_row[0]
        
        # Get role permissions
        cursor.execute('SELECT permissions FROM roles WHERE role_id = ?', (role,))
        permissions_row = cursor.fetchone()
        
        conn.close()
        
        if not permissions_row:
            return False
        
        permissions = json.loads(permissions_row[0])
        
        # Check if user has the permission
        return 'all' in permissions or permission in permissions
    
    def log_audit_event(self, user_id: Optional[str], action: str, resource: str,
                       details: Optional[Dict[str, Any]] = None, ip_address: str = "",
                       success: bool = True) -> int:
        """
        Log an audit event
        
        Args:
            user_id (Optional[str]): User ID
            action (str): Action performed
            resource (str): Resource affected
            details (Optional[Dict[str, any]]): Additional details
            ip_address (str): IP address
            success (bool): Whether the action was successful
            
        Returns:
            int: Log ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log (user_id, action, resource, details, ip_address, success)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id, action, resource,
            json.dumps(details) if details else None,
            ip_address, success
        ))
        
        log_id = cursor.lastrowid
        if log_id is None:
            log_id = 0
        conn.commit()
        conn.close()
        
        return log_id
    
    def get_audit_log(self, user_id: Optional[str] = None, action: Optional[str] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit log entries
        
        Args:
            user_id (Optional[str]): Filter by user ID
            action (Optional[str]): Filter by action
            limit (int): Maximum number of entries to return
            
        Returns:
            List[Dict[str, any]]: List of audit log entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        log_entries = []
        for row in rows:
            entry_data = dict(zip(column_names, row))
            # Parse JSON details
            if entry_data['details']:
                entry_data['details'] = json.loads(entry_data['details'])
            log_entries.append(entry_data)
        
        conn.close()
        return log_entries
    
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data
        
        Args:
            data (str): Data to encrypt
            
        Returns:
            str: Encrypted data (base64 encoded)
        """
        encrypted_data = self.cipher_suite.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data (str): Encrypted data (base64 encoded)
            
        Returns:
            str: Decrypted data
        """
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_data.decode('utf-8')
    
    def grant_access(self, resource_type: str, resource_id: str, user_id: str,
                    permission_level: str, granted_by: str) -> int:
        """
        Grant access to a resource
        
        Args:
            resource_type (str): Type of resource
            resource_id (str): Resource ID
            user_id (str): User ID
            permission_level (str): Permission level
            granted_by (str): User who granted the access
            
        Returns:
            int: Access control ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO access_control (resource_type, resource_id, user_id, permission_level, granted_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (resource_type, resource_id, user_id, permission_level, granted_by))
        
        ac_id = cursor.lastrowid
        if ac_id is None:
            ac_id = 0
        conn.commit()
        conn.close()
        
        return ac_id
    
    def check_resource_access(self, resource_type: str, resource_id: str, 
                            user_id: str, required_permission: str) -> bool:
        """
        Check if a user has access to a specific resource
        
        Args:
            resource_type (str): Type of resource
            resource_id (str): Resource ID
            user_id (str): User ID
            required_permission (str): Required permission level
            
        Returns:
            bool: True if user has access
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT permission_level FROM access_control
            WHERE resource_type = ? AND resource_id = ? AND user_id = ?
        ''', (resource_type, resource_id, user_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
        
        # In a real implementation, you would check permission hierarchy
        # For now, we'll do a simple check
        return result[0] == required_permission or result[0] == 'owner'

def create_sample_security_data(security_manager: SecurityManager) -> None:
    """
    Create sample security data for demonstration
    
    Args:
        security_manager (SecurityManager): Security manager instance
    """
    # Create sample users
    sample_users = [
        ("user001", "alice", "Alice123!", "reviewer", "alice@research.org"),
        ("user002", "bob", "Bob123!", "submitter", "bob@company.com"),
        ("user003", "charlie", "Charlie123!", "viewer", "charlie@university.edu")
    ]
    
    for user_id, username, password, role, email in sample_users:
        if not security_manager.user_exists(username):
            security_manager.create_user(user_id, username, password, role, email)
    
    # Grant sample access
    security_manager.grant_access("proposal", "PROP001", "user001", "review", "admin")
    security_manager.grant_access("proposal", "PROP002", "user002", "owner", "admin")

def main():
    """
    Main function to demonstrate the Security Manager
    """
    print("Demonstrating Security Manager for R&D Evaluation System")
    print("=" * 58)
    
    # Initialize security manager
    security = SecurityManager("demo_security.db")
    print("✓ Security Manager initialized")
    
    # Create sample data
    create_sample_security_data(security)
    print("✓ Sample security data created")
    
    # Demonstrate security features
    print("\nSecurity Features Demo:")
    
    # 1. User authentication
    print("\n1. User Authentication:")
    user_info = security.authenticate_user("admin", "Admin123!")
    if user_info:
        print(f"   Authenticated: {user_info['username']} ({user_info['role']})")
    else:
        print("   Authentication failed")
    
    # 2. Session management
    print("\n2. Session Management:")
    session_id = security.create_session("admin", "127.0.0.1", "Demo Browser")
    print(f"   Session created: {session_id[:10]}...")
    
    session_info = security.validate_session(session_id)
    if session_info:
        print(f"   Session valid for: {session_info['username']}")
    else:
        print("   Session invalid")
    
    # 3. Permission checking
    print("\n3. Permission Checking:")
    has_permission = security.has_permission("admin", "all")
    print(f"   Admin has 'all' permission: {has_permission}")
    
    has_permission = security.has_permission("user001", "evaluate")
    print(f"   Reviewer has 'evaluate' permission: {has_permission}")
    
    # 4. Data encryption
    print("\n4. Data Encryption:")
    sensitive_data = "This is sensitive proposal information"
    encrypted = security.encrypt_data(sensitive_data)
    print(f"   Original: {sensitive_data[:20]}...")
    print(f"   Encrypted: {encrypted[:20]}...")
    
    decrypted = security.decrypt_data(encrypted)
    print(f"   Decrypted: {decrypted[:20]}...")
    print(f"   Match: {sensitive_data == decrypted}")
    
    # 5. Audit logging
    print("\n5. Audit Logging:")
    log_id = security.log_audit_event(
        user_id="admin",
        action="login",
        resource="system",
        details={"ip": "127.0.0.1"},
        ip_address="127.0.0.1",
        success=True
    )
    print(f"   Audit event logged (ID: {log_id})")
    
    # Get recent audit logs
    audit_logs = security.get_audit_log(limit=5)
    print(f"   Recent audit events: {len(audit_logs)}")
    
    # 6. Access control
    print("\n6. Access Control:")
    access_granted = security.grant_access("proposal", "PROP003", "user002", "review", "admin")
    print(f"   Access granted (ID: {access_granted})")
    
    has_access = security.check_resource_access("proposal", "PROP003", "user002", "review")
    print(f"   User has access to proposal: {has_access}")
    
    print("\n" + "=" * 58)
    print("✓ Security Manager demonstration completed successfully")
    print(f"\nNote: Demo security database saved as 'demo_security.db'")
    print("Note: Encryption key saved as 'encryption.key'")

if __name__ == "__main__":
    main()